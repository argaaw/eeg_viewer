import os, tempfile, hashlib
import numpy as np
import requests
import streamlit as st
import h5py
from math import ceil

import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATASETS = {
    "DTU": {
        "id": "DTU",
        "fs": 64,
        "channels": 64,
        "subjects": 18,
        "trials": 60,
        "default_url": "https://huggingface.co/datasets/argaaw/dtu_eeg/resolve/main/eeg.h5",
        "grid_cols": 8,
    },
    "AVED": {
        "id": "AVED",
        "fs": 128,
        "channels": 32,
        "subjects": 20,
        "trials": 16,
        "default_url": "https://huggingface.co/datasets/argaaw/aved_eeg/resolve/main/eeg.h5",
        "grid_cols": 8,
    },
}

H5_KEY_TEMPLATE = "S{sub}/Tra{trial}"

st.set_page_config(page_title="EEG Dataset Viewer", layout="wide")
st.title("EEG Grid Viewer")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Dataset")

    dataset_name = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        index=0,
    )
    cfg = DATASETS[dataset_name]

    st.markdown(
        f"- **Channels**: {cfg['channels']}  \n"
        f"- **Default fs**: {cfg['fs']} Hz  \n"
        f"- **Subjects**: {cfg['subjects']}  \n"
        f"- **Trials**: {cfg['trials']}"
    )

    # Dropdown (Subject, Trial)
    subject = st.selectbox(
        "Subject", 
        list(range(1, cfg["subjects"] + 1)),
        index=0,
        format_func=lambda i: f"{i}"
    )
    trial = st.selectbox(
        "Trial",
        list(range(1, cfg["trials"] + 1)),
        index=0,
        format_func=lambda i: f"{i}")

    st.markdown("---")
    st.subheader("View options")
    zero_center = st.checkbox("Zero-mean per channel", value=False)
    share_y = st.checkbox("Share Y-scale across channels", value=False)
    line_width = st.slider("Line width", 0.5, 2.5, 1.0, 0.1)
    height_per_row = st.slider("Height per row (px)", 120, 260, 180, 10)
    show_titles = st.checkbox("Show channel titles", value=True)


# ---------------- Helpers ----------------
@st.cache_data(show_spinner=True)
def download_h5_to_tmp(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    tmp_path = os.path.join(tempfile.gettempdir(), f"eeg_{h}.h5")
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        return tmp_path
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_path


def load_h5(_path: str, _sub: int, _tr: int, expected_channels: int):
    if not os.path.exists(_path):
        raise FileNotFoundError(f"HDF5 file not found")
    
    key = H5_KEY_TEMPLATE.format(sub=_sub, trial=_tr)

    with h5py.File(_path, "r") as f:
        if key not in f:
            raise KeyError(f"Dataset key '{key}' not found in HDF5")
        
        ds = f[key][()]

    if ds.ndim != 2:
        raise ValueError(f"Expected shape (T, C), got {ds.shape} at '{key}'")

    T, C = ds.shape

    if C != expected_channels:
        raise ValueError(f"Expected channels = {expected_channels}, but got {C} at '{key}'")

    return ds, key


def plot_eeg_grid(
    arr: np.ndarray,
    fs: int,
    *,
    zero_center: bool,
    share_y: bool,
    line_width: float,
    height_per_row: int,
    show_titles: bool,
    grid_cols: int
):
    T, C = arr.shape
    t = np.arange(T) / float(fs)

    data = arr.astype(float)

    # NaN-safe zero-mean
    if zero_center:
        mu = np.nanmean(data, axis=0, keepdims=True)
        data = data - mu

    rows = ceil(C / grid_cols)
    cols = grid_cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
        subplot_titles=[f"Ch {i+1}" for i in range(C)] if show_titles else None,
    )

    # NaN-safe global y-range (optional)
    yr = None

    if share_y:
        finite = np.isfinite(data)

        if finite.any():
            ymin = float(np.nanmin(data))
            ymax = float(np.nanmax(data))
            margin = 0.02 * (ymax - ymin + 1e-9)
            yr = (ymin - margin, ymax + margin)
        else:
            yr = None

    # traces
    for ch in range(C):
        r = ch // cols + 1
        c = ch % cols + 1
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=data[:, ch],
                mode="lines",
                line=dict(width=line_width),
                name=f"Ch {ch+1}",
                hoverinfo="x+y+name",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(
            title_text="Time (s)" if r == rows else None,
            showgrid=False,
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text="Amplitude" if c == 1 else None,
            showgrid=False,
            showticklabels=(c == 1),
            row=r,
            col=c,
        )

    # Layout (no rangeslider here)
    fig.update_layout(
        height=height_per_row * rows,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )

    # Sync all x-axes
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(matches='x', row=r, col=c)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=rows, col=1)

    if yr is not None:
        fig.update_yaxes(range=list(yr))

    return fig

# ---------------- Main (reactive) ----------------
try:
    if not cfg["default_url"]:
        raise ValueError("This dataset has no default HDF5 URL configured.")
    
    path = download_h5_to_tmp(cfg["default_url"])

    arr, key = load_h5(path, subject, trial, expected_channels=cfg["channels"])

    # Status warning
    nan_ratio = np.isnan(arr).mean(axis=0)
    bad = [i+1 for i, r in enumerate(nan_ratio) if r > 0]
    if bad:
        st.warning(f"NaN 포함 채널: {bad}")

    st.caption(f"Loaded → Dataset: {cfg['id']} | Key: {key} | Sub={subject}, Trial={trial}")
    st.success(f"Data shape: {arr.shape} | fs: {cfg['fs']} Hz")

    fig = plot_eeg_grid(
        arr,
        cfg["fs"],
        zero_center=zero_center,
        share_y=share_y,
        line_width=line_width,
        height_per_row=height_per_row,
        show_titles=show_titles,
        grid_cols=cfg["grid_cols"]
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.info("사이드바 설정(데이터셋/Subject/Trial)을 확인하세요.")

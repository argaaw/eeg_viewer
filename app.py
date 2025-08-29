import os, tempfile, hashlib
import numpy as np
import requests
import streamlit as st
import h5py

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="DTU EEG preprocessed dataset", layout="wide")
st.title("EEG 8×8 Grid Viewer (HDF5)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data Settings")
    fs = st.number_input("Sampling rate (Hz)", min_value=1, max_value=4096, value=64, step=1)

    source = st.radio("HDF5 source", ["Remote URL", "Local path"], horizontal=True)
    if source == "Remote URL":
        h5_url = st.text_input("HDF5 URL", value="https://huggingface.co/datasets/argaaw/dtu_eeg/resolve/main/eeg.h5")
        h5_path_local = None
    else:
        h5_path_local = st.text_input("HDF5 file path", value=os.path.join("data", "eeg.h5"))
        h5_url = None

    # 드롭다운(Subject: 1..18, Trial: 1..60)
    subject = st.selectbox("Subject", list(range(1, 19)), index=0, format_func=lambda i: f"S{i}")
    trial = st.selectbox("Trial", list(range(1, 61)), index=0, format_func=lambda i: f"Tra{i}")

    st.markdown("---")
    st.subheader("Key template (fixed)")
    h5_key_template = st.text_input(
        "Dataset key template",
        value="S{sub}/Tra{trial}",
        help="Fixed: S{sub}/Tra{trial}"
    )

    st.markdown("---")
    st.subheader("View options")
    zero_center = st.checkbox("Zero-mean per channel (NaN-safe)", value=True)
    share_y = st.checkbox("Share Y-scale across channels (NaN-safe)", value=False)
    line_width = st.slider("Line width", 0.5, 2.5, 1.0, 0.1)
    height_per_row = st.slider("Height per row (px)", 120, 260, 180, 10)
    show_titles = st.checkbox("Show channel titles", value=True)

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=True)
def download_h5_to_tmp(url: str) -> str:
    """다운로드한 HDF5를 임시 파일에 캐시."""
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

def load_h5(_path: str, _key_tmpl: str, _sub: int, _tr: int):
    """HDF5에서 (T,64) 배열 로드 (형식 고정, fallback 없음)."""
    if not os.path.exists(_path):
        raise FileNotFoundError(f"HDF5 file not found: {_path}")
    key = _key_tmpl.format(sub=_sub, trial=_tr)
    with h5py.File(_path, "r") as f:
        if key not in f:
            raise KeyError(f"Dataset key '{key}' not found in {os.path.abspath(_path)}")
        ds = f[key][()]
    if ds.ndim != 2 or ds.shape[1] != 64:
        raise ValueError(f"Expected shape (T, 64), got {ds.shape} at '{key}'")
    return ds, f"HDF5: {os.path.abspath(_path)} :: {key}"

def plot_eeg_grid(arr: np.ndarray, fs: int, *, zero_center: bool, share_y: bool,
                  line_width: float, height_per_row: int, show_titles: bool):
    T, C = arr.shape
    assert C == 64, "Expected 64 channels"
    t = np.arange(T) / float(fs)

    data = arr.astype(float)
    # NaN-safe zero-mean
    if zero_center:
        mu = np.nanmean(data, axis=0, keepdims=True)
        data = data - mu

    fig = make_subplots(
        rows=8, cols=8, shared_xaxes=True,
        horizontal_spacing=0.01, vertical_spacing=0.02,
        subplot_titles=[f"Ch {i+1}" for i in range(64)] if show_titles else None,
    )

    # NaN-safe global y-range (optional)
    yr = None
    if share_y:
        ymin = float(np.nanmin(data))
        ymax = float(np.nanmax(data))
        margin = 0.02 * (ymax - ymin + 1e-9)
        yr = (ymin - margin, ymax + margin)

    # 64 traces
    for ch in range(64):
        r = ch // 8 + 1
        c = ch % 8 + 1
        fig.add_trace(
            go.Scattergl(
                x=t, y=data[:, ch],
                mode="lines",
                line=dict(width=line_width),
                name=f"Ch {ch+1}",
                hoverinfo="x+y+name",
                showlegend=False,
            ),
            row=r, col=c
        )
        fig.update_xaxes(
            title_text="Time (s)" if r == 8 else None,
            showgrid=False,
            row=r, col=c
        )
        fig.update_yaxes(
            title_text="µV" if c == 1 else None,
            showgrid=False,
            showticklabels=(c == 1),
            row=r, col=c
        )

    # Layout (no rangeslider here)
    fig.update_layout(
        height=height_per_row * 8,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )

    # Sync all x-axes
    for r in range(1, 9):
        for c in range(1, 9):
            fig.update_xaxes(matches='x', row=r, col=c)

    # Rangeslider only on bottom-left axis
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=8, col=1)

    if yr is not None:
        fig.update_yaxes(range=list(yr))

    return fig

# ---------------- Main (reactive) ----------------
try:
    # 원격 URL → 임시 파일로 캐시 후 사용
    if source == "Remote URL":
        if not h5_url or not h5_url.startswith(("http://", "https://")):
            raise ValueError("유효한 HDF5 URL을 입력하세요.")
        path = download_h5_to_tmp(h5_url)
    else:
        path = h5_path_local

    arr, info = load_h5(path, h5_key_template, subject, trial)

    # 상태 알림
    nan_ratio = np.isnan(arr).mean(axis=0)
    bad = [i+1 for i, r in enumerate(nan_ratio) if r > 0]
    if bad:
        st.warning(f"NaN 포함 채널: {bad}")

    st.caption(f"Loaded → {info}")
    st.success(f"Data shape: {arr.shape}")

    fig = plot_eeg_grid(
        arr, fs,
        zero_center=zero_center,
        share_y=share_y,
        line_width=line_width,
        height_per_row=height_per_row,
        show_titles=show_titles,
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.info("사이드바 설정과 HDF5 URL/경로, Key 템플릿(S{sub}/Tra{trial})을 확인하세요.")
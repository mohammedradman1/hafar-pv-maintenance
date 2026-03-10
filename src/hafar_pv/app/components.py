"""Reusable Streamlit components."""

from __future__ import annotations

import streamlit as st


def sidebar_info() -> None:
    """Render sidebar with project context."""

    st.sidebar.header("Hafar PV Maintenance")
    st.sidebar.markdown(
        "Segmentation + fault detection pipeline built with PyTorch."
    )


def upload_widget() -> None:
    """Render file uploader placeholder."""

    uploaded = st.file_uploader("Upload panel imagery", accept_multiple_files=False)
    if uploaded is not None:
        st.info("Inference pipeline is not wired yet. Placeholder only.")

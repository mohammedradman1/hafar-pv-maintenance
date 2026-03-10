"""Streamlit entry point for Hafar PV Maintenance."""

from __future__ import annotations

import streamlit as st

from ..config import get_settings
from . import components


def _render_header() -> None:
    st.title("Hafar PV Maintenance Dashboard")
    st.caption("Prototype interface for solar panel segmentation and fault detection.")


def _render_overview() -> None:
    st.subheader("Pipeline Overview")
    st.markdown(
        """
        1. **Segment** photovoltaic arrays to isolate individual panels.
        2. **Diagnose** each panel for defects (micro-cracks, hotspots, soiling).
        3. **Report** aggregated faults and confidence metrics.
        """
    )


def main() -> None:
    """Launch the Streamlit page."""

    st.set_page_config(page_title="Hafar PV Maintenance", layout="wide")
    components.sidebar_info()
    _render_header()
    _render_overview()
    st.divider()
    components.upload_widget()

    settings = get_settings()
    st.sidebar.metric("Data Root", str(settings.data_root))
    st.sidebar.metric("Models Root", str(settings.models_root))


if __name__ == "__main__":
    main()

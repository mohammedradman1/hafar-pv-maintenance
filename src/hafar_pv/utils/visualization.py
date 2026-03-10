"""Visualization helpers for panels and faults."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> plt.Figure:
    """Create a figure showing the mask overlayed on top of the image."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha, cmap="Reds")
    ax.axis("off")
    return fig

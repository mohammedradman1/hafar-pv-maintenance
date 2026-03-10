"""Command-line entry point to preprocess the Kaggle Photovoltaic Thermography dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hafar_pv.config import get_settings
from hafar_pv.data import PhotovoltaicThermographyPreprocessor
from hafar_pv.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_root",
        type=Path,
        help="Directory containing raw Kaggle files (images + JSON annotations).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination directory for processed tensors. Defaults to <data_root>/processed/photovoltaic-system-thermography.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of panels reserved for validation set.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of panels reserved for test set.",
    )
    parser.add_argument(
        "--resize-panels",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional size to resize panel crops (width height).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging(getattr(logging, args.log_level.upper()))

    settings = get_settings()
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(settings.data_root) / "processed" / "photovoltaic-system-thermography"
    )

    preprocessor = PhotovoltaicThermographyPreprocessor(
        raw_root=args.raw_root,
        output_root=output_root,
        resize_panels_to=tuple(args.resize_panels) if args.resize_panels else None,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()

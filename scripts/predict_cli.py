#!/usr/bin/env python
"""
Command‑line interface for the AgeGenderPredictor.

This script provides a simple way to generate sex and age predictions
from video titles supplied via a text file or directly as a command
line argument.  When reading from a file, each line is treated as
an independent title.  Predictions can be written to a CSV file or
displayed on standard output.

Usage
-----

Predict demographics for titles stored in a file and save the
results as CSV:

.. code-block:: bash

    python scripts/predict_cli.py --input path/to/titles.txt --output predictions.csv

Predict demographics for a single title provided directly on the
command line:

.. code-block:: bash

    python scripts/predict_cli.py --input "football match highlights"

The output will include numeric codes.  You can map these codes to
human‑readable labels (e.g. 0→male, 1→female) in downstream tools.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

# Ensure that the project source directory is on the Python path when
# running this script directly.  Without this adjustment the import
# below would fail with ``ModuleNotFoundError`` because ``src`` is
# not installed as a package.  In a production setting you would
# install the package (e.g. via `pip -e .`), but for a pet project
# adjusting sys.path suffices.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predictor import AgeGenderPredictor  # type: ignore


def _read_titles(input_arg: str) -> List[str]:
    """Interpret the input argument as a title or path to a file.

    If the path exists, lines are read from the file (skipping empty
    lines).  Otherwise the argument itself is treated as a single
    title.  This design allows for flexible usage without requiring
    separate flags.

    Parameters
    ----------
    input_arg: str
        Either a path to a text file containing one title per line
        or a literal title string.

    Returns
    -------
    list of str
        The extracted titles.
    """
    path = Path(input_arg)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            titles = [line.strip() for line in f if line.strip()]
    else:
        titles = [input_arg]
    return titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict user demographics from video titles.")
    parser.add_argument(
        "--input",
        required=True,
        help="A path to a text file with titles or a single title string.",
    )
    parser.add_argument(
        "--model-dir",
        default=Path(__file__).resolve().parents[1] / "models",
        help="Directory containing trained model files.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write predictions as CSV.  If omitted, results are printed to stdout.",
    )

    args = parser.parse_args()
    titles = _read_titles(args.input)
    predictor = AgeGenderPredictor(model_dir=args.model_dir)
    sex_preds, age_preds = predictor.predict(titles)
    df = pd.DataFrame({"title": titles, "sex": sex_preds, "age_class": age_preds})
    if args.output:
        out_path = Path(args.output)
        df.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
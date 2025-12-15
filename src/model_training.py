"""
Module for training age and sex prediction models on synthetic data.

This module defines a function to generate a small synthetic dataset
representative of video titles and associated demographic labels, and
then trains two separate Multinomial Naïve Bayes classifiers using a
TF‑IDF representation of the text.  The models along with their
fitted vectoriser are saved to disk in the ``models`` directory.  This
approach ensures that anyone cloning the repository can regenerate
compatible models without accessing the original proprietary data
used in the source project.

The training pipeline is intentionally simple to keep the example
self‑contained while demonstrating the overall architecture of
production‑ready machine learning code.  In practice you should
replace the synthetic data with your own labelled dataset and may
experiment with more sophisticated algorithms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def _generate_synthetic_dataset(n_samples: int = 200) -> pd.DataFrame:
    """Generate a synthetic dataset for demonstration purposes.

    Each record consists of a short video title and associated
    demographic labels: ``sex`` (0 for male, 1 for female) and
    ``age_class`` (0, 1 or 2 representing broad age groups).  The
    mapping between titles and labels encodes a few intuitive
    associations (e.g. sports content tends to skew towards males,
    beauty content towards females and cartoons towards younger
    viewers).  Random sampling with replacement is used to assemble
    the final dataset.

    Parameters
    ----------
    n_samples: int, optional
        The number of records to generate.  Defaults to 200.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``title``, ``sex`` and ``age_class``.
    """
    # Base examples encoding plausible relationships between
    # content and demographics.  The second element in the tuple is
    # the sex label, and the third is the age_class label.
    base_samples: List[Tuple[str, int, int]] = [
        ("football match highlights", 0, 1),        # male, adult
        ("makeup tutorial for beginners", 1, 1),     # female, adult
        ("cooking recipe quick pasta", 1, 1),        # female, adult
        ("video game gameplay walkthrough", 0, 0),   # male, youth
        ("technology news latest gadgets", 0, 1),    # male, adult
        ("romantic movie trailer", 1, 1),           # female, adult
        ("kids cartoon episode", 0, 0),             # male or female, youth
        ("political debate analysis", 0, 2),         # male, senior
        ("music concert live", 1, 1),                # female, adult
        ("science documentary space", 0, 2),         # male, senior
    ]
    titles, sexes, ages = zip(*base_samples)
    # Draw random indices with replacement to build a synthetic dataset.
    idx = np.random.choice(len(base_samples), size=n_samples, replace=True)
    synthetic_titles = [titles[i] for i in idx]
    synthetic_sexes = [sexes[i] for i in idx]
    synthetic_ages = [ages[i] for i in idx]
    return pd.DataFrame({
        "title": synthetic_titles,
        "sex": synthetic_sexes,
        "age_class": synthetic_ages,
    })


def train_models(
    output_dir: str | Path | None = None,
    n_samples: int = 200,
) -> Dict[str, Path]:
    """Train sex and age classifiers and save them to disk.

    Two MultinomialNB models are trained on a TF‑IDF representation
    of synthetic video titles.  Each model is saved as a dictionary
    containing both the estimator and the fitted vectoriser.  The
    output file names follow the convention ``sex_model.joblib`` and
    ``age_model.joblib``.

    Parameters
    ----------
    output_dir: str or pathlib.Path, optional
        Directory where the trained models will be saved.  It will be
        created if it does not exist.  Defaults to ``"models"``.

    n_samples: int, optional
        Number of synthetic training samples to generate.  Increase
        this value to produce larger training sets.  Defaults to
        200.

    Returns
    -------
    Dict[str, Path]
        A dictionary mapping model names (``"sex"`` and ``"age"``) to
        the paths of the saved joblib files.
    """
    # If no output directory is provided, default to the ``models``
    # folder alongside this module's parent directory (the project
    # root).  This ensures that models end up inside
    # ``VSR_pet_project/models`` when the module is executed via
    # ``python -m`` or imported.
    if output_dir is None:
        # The parent of this module's directory is the project root (VSR_pet_project).
        output_path = Path(__file__).resolve().parents[1] / "models"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate the dataset and shuffle it to remove any ordering bias.
    df = _generate_synthetic_dataset(n_samples)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Fit a single vectoriser on all titles.  In a real project you
    # would persist the vectoriser separately or share it between
    # models.  Here we store a copy alongside each classifier for
    # simplicity.
    vectoriser = TfidfVectorizer(stop_words="english")
    X = vectoriser.fit_transform(df["title"])

    # Train the sex classifier.
    sex_clf = MultinomialNB()
    sex_clf.fit(X, df["sex"])
    sex_path = output_path / "sex_model.joblib"
    dump({"model": sex_clf, "vectoriser": vectoriser}, sex_path)

    # Train the age classifier.
    age_clf = MultinomialNB()
    age_clf.fit(X, df["age_class"])
    age_path = output_path / "age_model.joblib"
    dump({"model": age_clf, "vectoriser": vectoriser}, age_path)

    return {"sex": sex_path, "age": age_path}


if __name__ == "__main__":
    # When executed as a script, train the models and report where
    # they were saved.  Using a modest number of samples keeps
    # training fast.
    model_paths = train_models()
    for name, path in model_paths.items():
        print(f"Saved {name} model to {path}")
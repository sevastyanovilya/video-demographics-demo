"""
Utilities for loading pre‑trained models and generating age and sex
predictions.

The :class:`AgeGenderPredictor` encapsulates two independent
MultinomialNB classifiers along with a shared TF‑IDF vectoriser.  It
provides a simple interface that accepts one or more video titles and
returns corresponding demographic predictions.  Loading models is
performed lazily on first use to keep initialisation lightweight.

Example
-------

.. code-block:: python

    from src.predictor import AgeGenderPredictor

    predictor = AgeGenderPredictor(model_dir="models")
    sex_preds, age_preds = predictor.predict([
        "football match highlights",
        "makeup tutorial for beginners",
    ])
    print(sex_preds)  # e.g. [0, 1]
    print(age_preds)  # e.g. [1, 1]

The numeric codes correspond to the synthetic labels used during
training (0 for male / youth etc.).  You can map them to more
interpretable categories in downstream code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from joblib import load


class AgeGenderPredictor:
    """Predict age classes and sex from video titles using pre‑trained models.

    Parameters
    ----------
    model_dir: str or pathlib.Path, optional
        Directory where the ``age_model.joblib`` and ``sex_model.joblib`` files
        reside.  If not provided, the current working directory is used.
    """

    def __init__(self, model_dir: str | Path = ".") -> None:
        self.model_dir = Path(model_dir)
        self._age_bundle = None
        self._sex_bundle = None

    def _load_models(self) -> None:
        """Lazy loader for the classifiers and vectoriser.

        Both classifiers are expected to be stored as dictionaries with
        keys ``"model"`` and ``"vectoriser"``.  The vectoriser from the
        sex model is reused for both predictions to ensure consistent
        feature space.
        """
        if self._sex_bundle is not None and self._age_bundle is not None:
            return
        sex_path = self.model_dir / "sex_model.joblib"
        age_path = self.model_dir / "age_model.joblib"
        if not sex_path.exists() or not age_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. Did you run the training?"
            )
        self._sex_bundle = load(sex_path)
        self._age_bundle = load(age_path)

        # Use the vectoriser from the sex bundle for both models.  This
        # assumes that the age model was trained with the same
        # vectoriser.  If you wish to store separate vectorisers you
        # can modify this assignment accordingly.
        self._vectoriser = self._sex_bundle["vectoriser"]

    def predict(self, titles: Iterable[str]) -> Tuple[List[int], List[int]]:
        """Predict sex and age classes for a sequence of video titles.

        The input sequence can be any iterable of strings.  Titles
        containing non‑ASCII characters are handled transparently.

        Parameters
        ----------
        titles: iterable of str
            Video titles or descriptions from which to infer demographics.

        Returns
        -------
        sex_predictions: list of int
            Predicted binary sex labels (0 for male, 1 for female).

        age_predictions: list of int
            Predicted age class labels (0, 1 or 2 corresponding to
            youth, adult and senior in the synthetic dataset).
        """
        # Ensure models and vectoriser are loaded.
        self._load_models()
        # Vectorise the input titles.  The vectoriser expects a list or
        # array‑like object and will handle tokenisation internally.
        X = self._vectoriser.transform(list(titles))
        sex_preds = self._sex_bundle["model"].predict(X)
        age_preds = self._age_bundle["model"].predict(X)
        return sex_preds.tolist(), age_preds.tolist()

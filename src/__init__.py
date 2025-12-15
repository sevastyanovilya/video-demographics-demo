"""Top‑level package for the demographic prediction project.

This package exposes the :class:`AgeGenderPredictor` for external use.
Importing directly from :mod:`src` makes it convenient to access
functionality without digging into internal modules.

Example
-------

.. code-block:: python

    from src import AgeGenderPredictor
    predictor = AgeGenderPredictor(model_dir="models")
    sex_preds, age_preds = predictor.predict(["example title"])
"""

from .predictor import AgeGenderPredictor  # noqa: F401

__all__ = ["AgeGenderPredictor"]
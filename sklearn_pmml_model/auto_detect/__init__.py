"""
The :mod:`sklearn.auto_detect` module implements methods to automatically
detect the type of model from a PMML file.
"""

# License: BSD 2-Clause

from .base import auto_detect_estimator, auto_detect_classifier, auto_detect_regressor

__all__ = [
  'auto_detect_estimator',
  'auto_detect_classifier',
  'auto_detect_regressor',
]

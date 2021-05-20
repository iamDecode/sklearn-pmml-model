"""
The :mod:`sklearn_pmml_model.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""

from .forest import PMMLForestClassifier
from .gb import PMMLGradientBoostingClassifier

__all__ = ["PMMLForestClassifier", "PMMLGradientBoostingClassifier"]

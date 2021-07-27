"""
The :mod:`sklearn_pmml_model.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""

# License: BSD 2-Clause

from .forest import PMMLForestClassifier, PMMLForestRegressor
from .gb import PMMLGradientBoostingClassifier, PMMLGradientBoostingRegressor

__all__ = [
    'PMMLForestClassifier',
    'PMMLForestRegressor',
    'PMMLGradientBoostingClassifier',
    'PMMLGradientBoostingRegressor'
]

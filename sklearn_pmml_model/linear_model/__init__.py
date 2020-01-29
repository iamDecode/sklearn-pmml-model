"""
The :mod:`sklearn_pmml_model.linear_model` module implements generalized linear models.
"""

from .implementations import PMMLLinearRegression, PMMLRidge, PMMLLasso, PMMLElasticNet

__all__ = ["PMMLLinearRegression", "PMMLRidge", "PMMLLasso", "PMMLElasticNet"]
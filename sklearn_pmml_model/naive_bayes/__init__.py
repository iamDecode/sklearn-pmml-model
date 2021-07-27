"""
The :mod:`sklearn_pmml_model.naive_bayes` module implements Naive Bayes
algorithms. These are supervised learning methods based on applying Bayes'
theorem with strong (naive) feature independence assumptions.
"""

# License: BSD 2-Clause

from .implementations import PMMLGaussianNB

__all__ = ['PMMLGaussianNB']

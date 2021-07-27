"""
The :mod:`sklearn_pmml_model.tree` module includes decision tree-based models for
classification and regression.
"""

# License: BSD 2-Clause

from .tree import PMMLTreeClassifier, PMMLTreeRegressor, get_tree, clone

__all__ = ['PMMLTreeClassifier', 'PMMLTreeRegressor', 'get_tree', 'clone']

"""
The :mod:`sklearn_pmml_model.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import PMMLTreeClassifier, get_tree, clone

__all__ = ["PMMLTreeClassifier", "get_tree", "clone"]
"""
The :mod:`sklearn_pmml_model.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import PMMLTreeClassifier, construct_tree

__all__ = ["PMMLTreeClassifier", "construct_tree"]
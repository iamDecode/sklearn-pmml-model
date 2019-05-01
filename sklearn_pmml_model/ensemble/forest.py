import numpy as np
import warnings
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE
from sklearn.ensemble import RandomForestClassifier
from sklearn_pmml_model.tree import PMMLBaseTreeEstimator
from sklearn.base import clone as _clone


class PMMLForestClassifier(PMMLBaseTreeEstimator, RandomForestClassifier):
  def __init__(self, pmml, field_labels=None, n_jobs=None):
    PMMLBaseTreeEstimator.__init__(self, pmml, field_labels=field_labels)

    mining_model = self.find(self.root, 'MiningModel')
    if mining_model is None:
      raise Exception('PMML model does not contain MiningModel.')

    segmentation = self.find(mining_model, 'Segmentation')
    if segmentation is None:
      raise Exception('PMML model does not contain Segmentation.')

    if segmentation.get('multipleModelMethod') != 'majorityVote':
      raise Exception('PMML model ensemble should use majority vote.')

    # Parse segments
    segments = self.findall(segmentation, 'Segment')
    valid_segments = [segment for segment in segments if self.find(segment, 'True') is not None]

    if len(valid_segments) < len(segments):
      warnings.warn("Warning: {} segment(s) ignored because of unsupported predicate.".format(len(segments) - len(valid_segments)))

    n_estimators = len(valid_segments)
    RandomForestClassifier.__init__(self, n_estimators=n_estimators, n_jobs=n_jobs)
    self._validate_estimator()

    clf = self._make_estimator(append=False, random_state=123)
    clf.classes_ = self.classes_
    clf.n_features_ = self.n_features_
    clf.n_outputs_ = self.n_outputs_
    clf.n_classes_ = self.n_classes_
    clf.n_categories = self.n_categories
    self.template_estimator = clf

    self.estimators_ = [self.get_tree(s) for s in valid_segments]

  def get_tree(self, segment):
    tree  = clone(self.template_estimator)

    tree_model = segment.find("TreeModel")

    if tree_model is None:
      raise Exception('PMML segment does not contain TreeModel.')

    if tree_model.get('splitCharacteristic') != 'binarySplit':
      raise Exception('Sklearn only supports binary tree models.')

    first_node = tree_model.find('Node')
    nodes, values = self.construct_tree(first_node)

    node_ndarray = np.ascontiguousarray(nodes, dtype=NODE_DTYPE)
    value_ndarray = np.ascontiguousarray(values)
    max_depth = None

    state = {
      'max_depth': (2 ** 31) - 1 if max_depth is None else max_depth,
      'node_count': node_ndarray.shape[0],
      'nodes': node_ndarray,
      'values': value_ndarray
    }
    tree.tree_.__setstate__(state)

    return tree

def clone(estimator):
  clone = _clone(estimator)
  clone.n_features_ = estimator.n_features_
  clone.n_outputs_ = estimator.n_outputs_
  clone.n_classes_ = estimator.n_classes_
  clone.n_categories = estimator.n_categories
  clone.tree_ = Tree(estimator.n_features_, np.asarray([estimator.n_classes_]), estimator.n_outputs_, estimator.n_categories)
  return clone
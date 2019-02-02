import numpy as np
import warnings
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE
from sklearn.ensemble import RandomForestClassifier
from sklearn_pmml_model.tree import PMMLBaseTreeEstimator

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

    trees = [self._make_estimator(append=False, random_state=123) for _ in range(n_estimators)]

    for i, segment in enumerate(valid_segments):
      tree = self.find(segment, "TreeModel")

      if tree is None:
        raise Exception('PMML segment does not contain TreeModel.')

      if tree.get('splitCharacteristic') != 'binarySplit':
        raise Exception('Sklearn only supports binary tree models.')

      tree_ = Tree(self.n_features_, np.array([self.n_classes_]), self.n_outputs_, self.n_categories)

      firstNode = self.find(tree, 'Node')
      nodes, values = self.construct_tree(firstNode)

      node_ndarray = np.ascontiguousarray(nodes, dtype=NODE_DTYPE)
      value_ndarray = np.ascontiguousarray(values)
      max_depth = None
      state = {
        'max_depth': (2 ** 31) - 1 if max_depth is None else max_depth,
        'node_count': node_ndarray.shape[0],
        'nodes': node_ndarray,
        'values': value_ndarray
      }

      tree_.__setstate__(state)

      trees[i].classes_ = self.classes_
      trees[i].n_features_ = self.n_features_
      trees[i].n_outputs_ = self.n_outputs_
      trees[i].n_classes_ = self.n_classes_
      trees[i].n_categories = self.n_categories
      trees[i].tree_ = tree_

    self.estimators_ = trees
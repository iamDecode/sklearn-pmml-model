import numpy as np
import warnings
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE
from sklearn.ensemble import RandomForestClassifier
from sklearn_pmml_model.base import PMMLBaseClassifier
from sklearn_pmml_model.tree import construct_tree
from sklearn.base import clone as _clone


class PMMLForestClassifier(PMMLBaseClassifier, RandomForestClassifier):
  """
  A random forest classifier.

  A random forest is a meta estimator that fits a number of decision tree
  classifiers on various sub-samples of the dataset and uses averaging to
  improve the predictive accuracy and control over-fitting.

  The PMML model consists out of a <Segmentation> element, that contains
  various <Segment> elements. Each segment contains it's own <TreeModel>.
  For Random Forests, only segments with a <True/> predicate are supported.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  n_jobs : int or None, optional (default=None)
      The number of jobs to run in parallel for the `predict` method.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors.

  See more
  --------
  http://dmg.org/pmml/v4-3/MultipleModels.html

  """
  def __init__(self, pmml, n_jobs=None):
    PMMLBaseClassifier.__init__(self, pmml)

    mining_model = self.root.find('MiningModel')
    if mining_model is None:
      raise Exception('PMML model does not contain MiningModel.')

    segmentation = mining_model.find('Segmentation')
    if segmentation is None:
      raise Exception('PMML model does not contain Segmentation.')

    if segmentation.get('multipleModelMethod') != 'majorityVote':
      raise Exception('PMML model ensemble should use majority vote.')

    # Parse segments
    segments = segmentation.findall('Segment')
    valid_segments = [segment for segment in segments if segment.find('True') is not None]

    if len(valid_segments) < len(segments):
      warnings.warn(
        'Warning: {} segment(s) ignored because of unsupported predicate.'
        .format(len(segments) - len(valid_segments))
      )

    n_estimators = len(valid_segments)
    RandomForestClassifier.__init__(self, n_estimators=n_estimators, n_jobs=n_jobs)
    self._validate_estimator()

    clf = self._make_estimator(append=False, random_state=123)
    clf.classes_ = self.classes_
    clf.n_features_ = self.n_features_
    clf.n_outputs_ = self.n_outputs_
    clf.n_classes_ = self.n_classes_
    self.template_estimator = clf

    self.estimators_ = [self.get_tree(s) for s in valid_segments]

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    for clf in self.estimators_:
      n_categories = np.asarray([
        len(self.field_mapping[field.get('name')][1].categories)
        if field.get('optype') == 'categorical' else -1
        for field in fields
        if field.tag == 'DataField'
      ], dtype=np.int32, order='C')
      clf.n_categories = n_categories
      clf.tree_.set_n_categories(n_categories)

  def get_tree(self, segment):
    """
    Method to train a single tree for a <Segment> PMML element.

    Parameters
    ----------
    segment : eTree.Element
        <Segment> element containing the decision tree to be imported.
        Only segments with a <True/> predicate are supported.

    Returns
    -------
    tree : sklearn.tree.DecisionTreeClassifier
        The sklearn decision tree instance imported from the provided segment.

    """
    tree = clone(self.template_estimator)

    tree_model = segment.find("TreeModel")

    if tree_model is None:
      raise Exception('PMML segment does not contain TreeModel.')

    if tree_model.get('splitCharacteristic') != 'binarySplit':
      raise Exception('Sklearn only supports binary tree models.')

    first_node = tree_model.find('Node')
    nodes, values = construct_tree(first_node, tree.classes_, self.field_mapping)

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


def clone(est, safe=True):
  """
  Helper method to clone a DecisionTreeClassifier, including private properties
  that are ignored in sklearn.base.clone.

  Parameters
  ----------
  est : BaseEstimator
      The estimator or group of estimators to be cloned.

  safe : boolean, optional
      If safe is false, clone will fall back to a deep copy on objects
      that are not estimators.

  """
  new_object = _clone(est, safe=safe)
  new_object.classes_ = est.classes_
  new_object.n_features_ = est.n_features_
  new_object.n_outputs_ = est.n_outputs_
  new_object.n_classes_ = est.n_classes_
  new_object.tree_ = Tree(est.n_features_, np.asarray([est.n_classes_]),
                          est.n_outputs_, np.array([], dtype=np.int32))
  return new_object

# License: BSD 2-Clause

import numpy as np
import struct
from sklearn.base import clone as _clone
from sklearn_pmml_model.base import PMMLBaseClassifier, PMMLBaseRegressor, parse_array
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_pmml_model.datatypes import Category
from operator import add
from warnings import warn
from xml.etree import cElementTree as eTree

SPLIT_UNDEFINED = struct.pack('d', TREE_UNDEFINED)


class PMMLTreeClassifier(PMMLBaseClassifier, DecisionTreeClassifier):
  """
  A decision tree classifier.

  The PMML model consists out of a <TreeModel> element, containing at least one
  <Node> element. Every node element contains a predicate, and optional <Node>
  children. Leaf nodes either have a score attribute or <ScoreDistribution>
  child describing the classification output.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/TreeModel.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)

    tree_model = self.root.find('TreeModel')

    if tree_model is None:
      raise Exception('PMML model does not contain TreeModel.')

    # Parse tree
    try:
      self.tree_ = Tree(self.n_features_in_, np.array([self.n_classes_], dtype=np.intp),
                        self.n_outputs_, np.array([], dtype=np.int32))
    except AttributeError:
      self.tree_ = Tree(self.n_features_, np.array([self.n_classes_], dtype=np.intp),
                        self.n_outputs_, np.array([], dtype=np.int32))

    split = tree_model.get('splitCharacteristic')
    if split == 'binarySplit':
      first_node = tree_model.find('Node')
    else:
      first_node = unflatten(tree_model.find('Node'))

    nodes, values = construct_tree(first_node, self.classes_, self.field_mapping)

    node_ndarray = np.ascontiguousarray(nodes, dtype=NODE_DTYPE)
    value_ndarray = np.ascontiguousarray(values)
    max_depth = None

    state = {
      'max_depth': (2 ** 31) - 1 if max_depth is None else max_depth,
      'node_count': node_ndarray.shape[0],
      'nodes': node_ndarray,
      'values': value_ndarray
    }
    self.tree_.__setstate__(state)

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    n_categories = np.asarray([
      len(self.field_mapping[field.get('name')][1].categories)
      if field.get('optype') == 'categorical' else -1
      for field in fields
      if field.tag == 'DataField'
    ], dtype=np.int32, order='C')

    self.tree_.set_n_categories(n_categories)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return DecisionTreeClassifier._more_tags(self)


class PMMLTreeRegressor(PMMLBaseRegressor, DecisionTreeRegressor):
  """
  A decision tree regressor.

  The PMML model consists out of a <TreeModel> element, containing at least one
  <Node> element. Every node element contains a predicate, and optional <Node>
  children. Leaf nodes either have a score attribute or <ScoreDistribution>
  child describing the classification output.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/TreeModel.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)

    tree_model = self.root.find('TreeModel')

    if tree_model is None:
      raise Exception('PMML model does not contain TreeModel.')

    # Parse tree
    self.n_outputs_ = 1
    n_classes = np.array([1] * self.n_outputs_, dtype=np.intp)
    try:
      self.tree_ = Tree(self.n_features_in_, n_classes, self.n_outputs_,
                        np.array([], dtype=np.int32))
    except AttributeError:
      self.tree_ = Tree(self.n_features_, n_classes, self.n_outputs_,
                        np.array([], dtype=np.int32))

    split = tree_model.get('splitCharacteristic')
    if split == 'binarySplit':
      first_node = tree_model.find('Node')
    else:
      first_node = unflatten(tree_model.find('Node'))

    nodes, values = construct_tree(first_node, None, self.field_mapping, rescale_factor=0.1)

    node_ndarray = np.ascontiguousarray(nodes, dtype=NODE_DTYPE)
    value_ndarray = np.ascontiguousarray(values)
    max_depth = None

    state = {
      'max_depth': (2 ** 31) - 1 if max_depth is None else max_depth,
      'node_count': node_ndarray.shape[0],
      'nodes': node_ndarray,
      'values': value_ndarray
    }
    self.tree_.__setstate__(state)

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    n_categories = np.asarray([
      len(self.field_mapping[field.get('name')][1].categories)
      if field.get('optype') == 'categorical' else -1
      for field in fields
      if field.tag == 'DataField'
    ], dtype=np.int32, order='C')

    self.tree_.set_n_categories(n_categories)

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return DecisionTreeRegressor._more_tags(self)


def unflatten(node):
  """
  Convert a `multiSplit` into a `binarySplit` decision tree which is expressively equivalent.

  Parameters
  ----------
  node : eTree.Element
      XML Node element representing the current node.

  Returns
  -------
  node : eTree.Element
    Modified XML Node element representing the flattened decision tree.

  """
  child_nodes = node.findall('Node')
  child_nodes = [
    node for node in child_nodes
    if getattr(node.find('SimplePredicate'), 'attrib', {}).get('operator') != 'isMissing'  # filter isMissing nodes
  ]

  parent = node
  for child in child_nodes:
    new_node = eTree.Element('Node')
    new_node.append(eTree.Element('True'))
    new_node.set('score', parent.get('score', 0))
    predicate = [e for e in parent if e.tag != 'Node']
    left_child = unflatten(child)

    if left_child.find('True') is not None and left_child.find('Node') is None:  # leaf node
      parent[:] = left_child[:]
      parent.attrib = left_child.attrib
    else:
      parent[:] = [*predicate, left_child, new_node]

    parent = new_node

  return node


def construct_tree(node, classes, field_mapping, i=0, rescale_factor=1):
  """
  Generate nodes and values used for constructing Cython Tree class.

  Parameters
  ----------
  node : eTree.Element
      XML Node element representing the current node.

  classes : list, None
      List of possible target classes. Is `None` for regression trees.

  field_mapping: { str: (int, callable) }
      Dictionary mapping column names to tuples with 1) index of the column and
      2) type of the column.

  i : int
      Index of the node in the result list.

  rescale_factor : float
      Factor to scale the output of every node with. Required for gradient
      boosting trees. Optional, and 1 by default.

  Returns
  -------
  (nodes, values) : tuple

      nodes : [()]
          List of nodes represented by: left child (int), right child (int),
          feature (int), value (int for categorical, float for continuous),
          impurity (float), sample count (int) and weighted sample count (int).

      values : [[]]
          List with training sample distributions at this node in the tree.

  """
  child_nodes = node.findall('Node')
  impurity = 0  # TODO: impurity doesnt affect predictions, but is nice to have
  i += 1

  def votes_for(field):
    # Deal with case where target field is a double, but ScoreDistribution value is an integer.
    if isinstance(field, float) and field.is_integer():
      return node.find(f"ScoreDistribution[@value='{field}']") or node.find(f"ScoreDistribution[@value='{int(field)}']")

    return node.find(f"ScoreDistribution[@value='{field}']")

  if not child_nodes:
    record_count = node.get('recordCount')

    if record_count is not None and classes is not None:
      node_count_weighted = float(record_count)
      node_count = int(node_count_weighted)
      votes = [[[float(votes_for(c).get('recordCount')) if votes_for(c) is not None else 0.0 for c in classes]]]
    else:
      score = node.get('score')
      node_count, node_count_weighted = (0, 0.0)

      if classes is None:
        # FIXME: unsure about `10 x rescale_factor`, but seems required, at least for r2pmml generated models
        votes = [[[float(score) * 10 * rescale_factor]]]
      else:
        votes = [[[1.0 if str(c) == score else 0.0 for c in classes]]]

    return [(TREE_LEAF, TREE_LEAF, TREE_UNDEFINED, SPLIT_UNDEFINED, impurity,
             node_count, node_count_weighted)], votes

  predicate = child_nodes[0].find('SimplePredicate')
  set_predicate = child_nodes[0].find('SimpleSetPredicate')

  # Convert SimplePredicate with equals operator on category to set predicate
  if predicate is not None:
    is_categorical = isinstance(field_mapping[predicate.get('field')][1], Category)

    if predicate.get('operator') == 'equal' and is_categorical:
      set_predicate = eTree.fromstring(f'''
      <SimpleSetPredicate field="{predicate.get('field')}" booleanOperator="isIn">
       <Array type="string">&quot;{predicate.get('value')}&quot;</Array>
      </SimpleSetPredicate>
      ''')
      predicate = None
    elif predicate.get('operator') == 'notEqual' and is_categorical:
      set_predicate = eTree.fromstring(f'''
      <SimpleSetPredicate field="{predicate.get('field')}" booleanOperator="isNotIn">
       <Array type="string">&quot;{predicate.get('value')}&quot;</Array>
      </SimpleSetPredicate>
      ''')
      predicate = None

  if predicate is not None and predicate.get('operator') in ['greaterThan', 'greaterOrEqual']:
    child_nodes.reverse()

  left_node, left_value = construct_tree(child_nodes[0], classes, field_mapping, i, rescale_factor)
  offset = len(left_node)
  right_node, right_value = construct_tree(child_nodes[1], classes, field_mapping, i + offset,
                                           rescale_factor)

  children = left_node + right_node
  distributions = left_value + right_value

  if predicate is not None:
    column, _ = field_mapping[predicate.get('field')]

    # We do not use field_mapping type as the Cython tree only supports floats
    value = np.float64(predicate.get('value'))

    # Account for `>=` != `>` and `<` != `<=`. scikit-learn only uses `<=`.
    if predicate.get('operator') == 'greaterOrEqual':
      value = np.nextafter(value, value - 1)
    if predicate.get('operator') == 'lessThan':
      value = np.nextafter(value, value - 1)
  else:
    if set_predicate is not None:
      column, field_type = field_mapping[set_predicate.get('field')]

      array = set_predicate.find('Array')
      categories = parse_array(array)

      mask = 0

      for category in categories:
        try:
          index = field_type.categories.index(category)
          mask |= 1 << index
        except ValueError:
          warn('Categorical values are missing in the PMML document, '
               'attempting to infer from decision tree splits.')
          field_type.categories.append(category)
          mask |= 1 << len(field_type.categories) - 1

      value = struct.pack('Q', np.uint64(mask))  # Q = unsigned long long = uint64

      if set_predicate.get('booleanOperator') == 'isNotIn':
        value = struct.pack('Q', ~np.uint64(mask))
    else:
      raise Exception('Unsupported tree format: unknown predicate structure in Node {}'
                      .format(child_nodes[0].get('id')))

  if classes is None:
    distribution = [[0]]
    sample_count_weighted = 0
    sample_count = 0
  else:
    distribution = [list(map(add, distributions[0][0], distributions[offset][0]))]
    sample_count_weighted = sum(distribution[0])
    sample_count = int(sample_count_weighted)

  return [(i, i + offset, column, value, impurity, sample_count, sample_count_weighted)] + children, \
         [distribution] + distributions


def get_tree(est, segment, rescale_factor=1) -> object:
  """
  Construct a single tree for a <Segment> PMML element.

  Parameters
  ----------
  est:
      The estimator to built the tree for. Should contain `template_estimator` and
      `field_mapping` attributes.

  segment : eTree.Element
      <Segment> element containing the decision tree to be imported.
      Only segments with a <True/> predicate are supported.

  rescale_factor : float
      Factor to scale the output of every node with. Required for gradient
      boosting trees. Optional, and 1 by default.

  Returns
  -------
  tree : sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor
      The sklearn decision tree instance imported from the provided segment,
      matching the type specified in est.template_estimator.

  """
  tree = clone(est.template_estimator)

  tree_model = segment.find('TreeModel')

  if tree_model is None:
    raise Exception('PMML segment does not contain TreeModel.')

  split = tree_model.get('splitCharacteristic')
  if split == 'binarySplit':
    first_node = tree_model.find('Node')
  else:
    first_node = unflatten(tree_model.find('Node'))

  if isinstance(tree, DecisionTreeClassifier):
    nodes, values = construct_tree(first_node, tree.classes_, est.field_mapping, rescale_factor=rescale_factor)
    node_ndarray = np.ascontiguousarray(nodes, dtype=NODE_DTYPE)
  else:
    nodes, values = construct_tree(first_node, None, est.field_mapping, rescale_factor=rescale_factor)
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
  Clone a DecisionTree, including private properties that are ignored in sklearn.base.clone.

  Parameters
  ----------
  est : BaseEstimator
      The estimator or group of estimators to be cloned.

  safe : boolean, optional
      If safe is false, clone will fall back to a deep copy on objects
      that are not estimators.

  """
  new_object = _clone(est, safe=safe)
  try:
    new_object.n_features_in_ = est.n_features_in_
  except AttributeError:
    new_object.n_features_ = est.n_features_
  new_object.n_outputs_ = est.n_outputs_

  if isinstance(est, DecisionTreeClassifier):
    new_object.classes_ = est.classes_
    new_object.n_classes_ = est.n_classes_
    n_classes = np.asarray([est.n_classes_], dtype=np.intp)

    try:
      new_object.tree_ = Tree(est.n_features_in_, n_classes, est.n_outputs_,
                              np.array([], dtype=np.int32))
    except AttributeError:
      new_object.tree_ = Tree(est.n_features_, n_classes, est.n_outputs_,
                              np.array([], dtype=np.int32))
  else:
    n_classes = np.array([1] * est.n_outputs_, dtype=np.intp)
    try:
      new_object.tree_ = Tree(est.n_features_in_, n_classes, est.n_outputs_,
                              np.array([], dtype=np.int32))
    except AttributeError:
      new_object.tree_ = Tree(est.n_features_, n_classes, est.n_outputs_,
                              np.array([], dtype=np.int32))

  return new_object

import numpy as np
import re
import struct
from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn_pmml_model.datatypes import PMMLDType, Interval
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier


class PMMLBaseTreeEstimator(PMMLBaseEstimator):
  def __init__(self, pmml, field_labels=None):
    super().__init__(pmml, field_labels=field_labels)

    self.classes_ = np.array([
      self.parse_type(e.get('value'), self.target_field, force_native=True)
      for e in self.findall(self.target_field, 'Value')
    ])

    target = self.target_field.get('name')
    fields = [ field for name, field in self.fields.items() if name != target ]

    if field_labels is not None:
      self.n_features_ = len(field_labels)
    else:
      self.n_features_ = len([field for field in fields if field.tag == f'{{{self.namespace}}}DataField'])

    self.n_outputs_ = 1
    self.n_classes_ = len(self.classes_)
    self.n_categories = np.array([
      len(self.findall(field, "Value")) if field.get('optype') == 'categorical' else -1
      for field in fields
    ]).astype(np.int32)

  def construct_tree(self, node, i = 0):
    """
    Generator for nodes and values used for constructing cython Tree class.

    Parameters
    ----------
    node : eTree.Element
        XML Node element representing the current node.

    i : int
        Index of the node in the result list.

    Returns
    -------
    nodes : [[int, int, int, Any, float, int, int]]
        List of nodes represented by: left child, right child, feature, value, impurity, sample count and
        sample count weighted.

    values : [[]]
        List with the distribution of training samples at this node in the tree.

    """
    childNodes = self.findall(node, 'Node')

    if len(childNodes) == 0:
      impurity = 0 # TODO: impurity is not really required for anything, but would be nice to have

      if node.get('recordCount') is not None:
        node_count = int(float(node.get('recordCount')))
        node_count_weighted = float(node.get('recordCount'))
        votes = np.array([[[float(e.get('recordCount')) for e in self.findall(node, 'ScoreDistribution')]]])
      else:
        node_count, node_count_weighted = (0, float(0.0))
        votes = np.array([[[float(1) if str(c) == node.get('score') else float(0) for c in self.classes_]]])

      return [
        [(TREE_LEAF, TREE_LEAF, TREE_UNDEFINED, struct.pack('d', TREE_UNDEFINED), impurity, node_count, node_count_weighted)],
        votes
      ]

    left_node, left_value = self.construct_tree(childNodes[0], i + 1)
    offset = len(left_node)
    right_node, right_value = self.construct_tree(childNodes[1], i + 1 + offset)

    children = left_node + right_node
    distributions = np.concatenate((left_value, right_value))

    predicate = self.find(childNodes[0], 'SimplePredicate')
    set_predicate = self.find(childNodes[0], 'SimpleSetPredicate')

    if predicate is not None:
      column, mapping = self.field_mapping[predicate.get('field')]
      value = mapping(predicate.get('value'))

      if isinstance(value, PMMLDType):
        value = value.value

      value = struct.pack('d', float(value)) # d = double = float64
    elif set_predicate is not None:
      column, mapping = self.field_mapping[set_predicate.get('field')]

      array = self.find(set_predicate, 'Array')
      values = [re.sub(r"(?<!\\)\"", '', value).replace('\"', '"') for value in array.text.split()]
      categories = [mapping(value) for value in values]

      bitmask = 0
      for category in categories:
        bitmask |= 1 << (category.categories.index(category))

      value = struct.pack('Q', bitmask) # Q = unsigned long long = uint64

      if set_predicate.get('booleanOperator') == 'isNotIn':
        value = struct.pack('Q', ~np.uint64(bitmask))
    else:
      raise Exception("Unsupported tree format: unknown predicate structure in Node {}".format(childNodes[0].get('id')))

    impurity = 0 # TODO: impurity is not really required for anything, but would be nice to have

    distributions_children = distributions[[0, offset]]
    distribution = np.sum(distributions_children, axis=0)
    sample_count = int(np.sum(distribution))
    sample_count_weighted = float(np.sum(distribution))

    return [(i + 1, i + 1 + offset, column, value, impurity, sample_count, sample_count_weighted)] + children, \
           np.concatenate((np.array([distribution]), distributions))


class PMMLTreeClassifier(PMMLBaseTreeEstimator, DecisionTreeClassifier):
    def __init__(self, pmml, field_labels=None):
      super().__init__(pmml, field_labels=field_labels)

      self.tree = self.find(self.root, 'TreeModel')

      if self.tree is None:
        raise Exception('PMML model does not contain TreeModel.')

      if self.tree.get('splitCharacteristic') != 'binarySplit':
        raise Exception('Sklearn only supports binary tree models.')

      self.tree_ = Tree(self.n_features_, np.array([self.n_classes_]), self.n_outputs_, self.n_categories)

      firstNode = self.find(self.tree, 'Node')
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
      self.tree_.__setstate__(state)
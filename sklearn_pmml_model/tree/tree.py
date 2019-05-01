import numpy as np
import struct
from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier
from operator import add

SPLIT_UNDEFINED = struct.pack('d', TREE_UNDEFINED)

class PMMLBaseTreeEstimator(PMMLBaseEstimator):
  def __init__(self, pmml, field_labels=None):
    super().__init__(pmml, field_labels=field_labels)

    target_type = self.get_type(self.target_field)
    self.classes_ = np.array([
      target_type(e.get('value'))
      for e in self.findall(self.target_field, 'Value')
    ])

    target = self.target_field.get('name')
    fields = [ field for name, field in self.fields.items() if name != target ]

    if field_labels is not None:
      self.n_features_ = len(field_labels)
    else:
      self.n_features_ = len([field for field in fields if field.tag == 'DataField'])

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
    impurity = 0 # TODO: impurity is not really required for anything, but would be nice to have
    i += 1

    if not childNodes:
      record_count = node.get('recordCount')

      if record_count is not None:
        node_count_weighted = float(record_count)
        node_count = int(node_count_weighted)
        votes = [[[float(e.get('recordCount')) for e in node.findall('ScoreDistribution')]]]
      else:
        score = node.get('score')

        if score is not None:
          node_count, node_count_weighted = (0, 0.0)
          votes = [[[1.0 if str(c) == score else 0.0 for c in self.classes_]]]
        else:
          raise Exception("Node has insufficient information to determine score: recordCount or score attributed expected")

      return [(TREE_LEAF, TREE_LEAF, TREE_UNDEFINED, SPLIT_UNDEFINED, impurity, node_count, node_count_weighted)], \
             votes

    left_node, left_value = self.construct_tree(childNodes[0], i)
    offset = len(left_node)
    right_node, right_value = self.construct_tree(childNodes[1], i + offset)

    children = left_node + right_node
    distributions = left_value + right_value

    predicate = self.find(childNodes[0], 'SimplePredicate')

    if predicate is not None:
      column, _ = self.field_mapping[predicate.get('field')]
      value = predicate.get('value') # We do not use field_mapping type as the cython tree only supports floats
      value = struct.pack('d', float(value)) # d = double = float64
    else:
      set_predicate = self.find(childNodes[0], 'SimpleSetPredicate')

      if set_predicate is not None:
        column, type = self.field_mapping[set_predicate.get('field')]

        array = set_predicate.find('Array')
        categories = [value.replace('\\"', '▲').replace('"', '').replace('▲', '"') for value in array.text.split()]

        mask = 0
        for category in categories:
          mask |= 1 << (type.categories.index(category))

        value = struct.pack('Q', mask) # Q = unsigned long long = uint64

        if set_predicate.get('booleanOperator') == 'isNotIn':
          value = struct.pack('Q', ~np.uint64(mask))
      else:
        raise Exception("Unsupported tree format: unknown predicate structure in Node {}".format(childNodes[0].get('id')))

    distribution = [list(map(add, distributions[0][0], distributions[offset][0]))]
    sample_count_weighted = sum(distribution[0])
    sample_count = int(sample_count_weighted)

    return [(i, i + offset, column, value, impurity, sample_count, sample_count_weighted)] + children, \
           [distribution] + distributions


class PMMLTreeClassifier(PMMLBaseTreeEstimator, DecisionTreeClassifier):
  def __init__(self, pmml, field_labels=None):
    super().__init__(pmml, field_labels=field_labels)

    tree_model = self.root.find('TreeModel')

    if tree_model is None:
      raise Exception('PMML model does not contain TreeModel.')

    if tree_model.get('splitCharacteristic') != 'binarySplit':
      raise Exception('Sklearn only supports binary tree models.')

    self.tree_ = Tree(self.n_features_, np.array([self.n_classes_]), self.n_outputs_, self.n_categories)

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
    self.tree_.__setstate__(state)
import numpy as np
from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier

class PMMLTreeClassifier(PMMLBaseEstimator, DecisionTreeClassifier):
  def __init__(self, pmml, field_labels=None):
    super().__init__(pmml, field_labels=field_labels)

    self.tree = self.find(self.root, 'TreeModel')

    if self.tree is None:
      raise Exception('PMML model does not contain TreeModel.')

    if self.tree.get('splitCharacteristic') != 'binarySplit':
      raise Exception('Sklearn only supports binary classification models.')

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
    self.tree_ = Tree(self.n_features_, np.array([self.n_classes_]), self.n_outputs_)

    firstNode = self.find(self.tree, 'Node')
    nodes, values = self.constructTree(firstNode)

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

  def constructTree(self, node, i = 0):
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
      node_count = int(float(node.get('recordCount')))
      node_count_weighted = float(node.get('recordCount'))

      return [
        [(TREE_LEAF, TREE_LEAF, TREE_UNDEFINED, TREE_UNDEFINED, impurity, node_count, node_count_weighted)],
        np.array([[[float(e.get('recordCount')) for e in self.findall(node, 'ScoreDistribution')]]])
      ]

    left_node, left_value = self.constructTree(childNodes[0], i + 1)
    offset = len(left_node)
    right_node, right_value = self.constructTree(childNodes[1], i + 1 + offset)

    children = left_node + right_node
    distributions = np.concatenate((left_value, right_value))

    predicate = self.find(childNodes[0], 'SimplePredicate')
    column, mapping = self.field_mapping[predicate.get('field')]
    value = mapping(predicate.get('value'))
    impurity = 0 # TODO: impurity is not really required for anything, but would be nice to have

    distributions_children = distributions[[0, offset]]
    distribution = np.sum(distributions_children, axis=0)
    sample_count = int(np.sum(distribution))
    sample_count_weighted = float(np.sum(distribution))

    return [(i + 1, i + 1 + offset, column, value, impurity, sample_count, sample_count_weighted)] + children, \
           np.concatenate((np.array([distribution]), distributions))
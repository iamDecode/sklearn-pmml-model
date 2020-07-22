import numpy as np
import struct
from sklearn_pmml_model.base import PMMLBaseClassifier
from sklearn_pmml_model.tree._tree import Tree, NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier
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


def unflatten(node):
  """
  This method converts a `mutliSplit` decision tree into a `binarySplit`
  decision tree which is expressively equivalent.

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

  parent = node
  for child in child_nodes:
    new_node = eTree.Element("Node")
    new_node.append(eTree.Element("True"))
    predicate = [e for e in parent if e.tag != 'Node']
    left_child = unflatten(child)

    if left_child.find('True') is not None and left_child.find('Node') is None:  # leaf node
      parent[:] = left_child[:]
      parent.attrib = left_child.attrib
    else:
      parent[:] = [*predicate, left_child, new_node]

    parent = new_node

  return node


def construct_tree(node, classes, field_mapping, i=0):
  """
  Generator for nodes and values used for constructing cython Tree class.

  Parameters
  ----------
  node : eTree.Element
      XML Node element representing the current node.

  classes : list
      List of possible target classes.

  field_mapping: { str: (int, callable) }
      Dictionary mapping column names to tuples with 1) index of the column and
      2) type of the column.

  i : int
      Index of the node in the result list.

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

  if not child_nodes:
    record_count = node.get('recordCount')

    if record_count is not None:
      node_count_weighted = float(record_count)
      node_count = int(node_count_weighted)

      def votesFor(target):
        # Deal with case where targetfield a double, but ScoreDistribution value
        # is an integer.
        if isinstance(target, float) and target.is_integer():
          return node.find(f"ScoreDistribution[@value='{target}']") or node.find(f"ScoreDistribution[@value='{int(target)}']")

        return node.find(f"ScoreDistribution[@value='{target}']")

      votes = [[[float(votesFor(c).get('recordCount')) if votesFor(c) is not None else 0.0 for c in classes]]]
    else:
      score = node.get('score')

      if score is not None:
        node_count, node_count_weighted = (0, 0.0)
        votes = [[[1.0 if str(c) == score else 0.0 for c in classes]]]
      else:
        raise Exception('Node has insufficient information to determine output:'
                        + ' recordCount or score attributes expected')

    return [(TREE_LEAF, TREE_LEAF, TREE_UNDEFINED, SPLIT_UNDEFINED, impurity,
             node_count, node_count_weighted)], votes

  left_node, left_value = construct_tree(child_nodes[0], classes, field_mapping, i)
  offset = len(left_node)
  right_node, right_value = construct_tree(child_nodes[1], classes, field_mapping, i + offset)

  children = left_node + right_node
  distributions = left_value + right_value

  predicate = child_nodes[0].find('SimplePredicate')

  if predicate is not None:
    column, _ = field_mapping[predicate.get('field')]
    # We do not use field_mapping type as the Cython tree only supports floats
    value = predicate.get('value')
    value = struct.pack('d', float(value))  # d = double = float64
  else:
    set_predicate = child_nodes[0].find('SimpleSetPredicate')

    if set_predicate is not None:
      column, field_type = field_mapping[set_predicate.get('field')]

      array = set_predicate.find('Array')
      categories = [
        value.replace('\\"', '▲').replace('"', '').replace('▲', '"')
        for value in array.text.split()
      ]

      mask = 0

      for category in categories:
        try:
          index = field_type.categories.index(category)
          mask |= 1 << index
        except ValueError:
          warn('Categorical values are missing in the PMML document, '
               + 'attempting to infer from decision tree splits.')
          field_type.categories.append(category)
          mask |= 1 << len(field_type.categories) - 1

      value = struct.pack('Q', mask)  # Q = unsigned long long = uint64

      if set_predicate.get('booleanOperator') == 'isNotIn':
        value = struct.pack('Q', ~np.uint64(mask))
    else:
      raise Exception("Unsupported tree format: unknown predicate structure in Node {}"
                      .format(child_nodes[0].get('id')))

  distribution = [list(map(add, distributions[0][0], distributions[offset][0]))]
  sample_count_weighted = sum(distribution[0])
  sample_count = int(sample_count_weighted)

  return [(i, i + offset, column, value, impurity, sample_count, sample_count_weighted)] + children, \
         [distribution] + distributions

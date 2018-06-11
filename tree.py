import numpy as np
from base import *


class PMMLTreeClassifier(PMMLBaseEstimator):
  def __init__(self, pmml):
    super(PMMLTreeClassifier, self).__init__(pmml)

    self.tree = self.find(self.root, 'TreeModel')

    if self.tree is None:
      raise Exception("PMML model does not contain TreeModel.")


  def evaluate_node(self, node, instance):
    predicates = {
      'True': lambda *_: True,
      'False': lambda *_: False,
      'SimplePredicate': self.evaluate_simple_predicate,
      'CompoundPredicate': lambda *_: (_ for _ in ()).throw(Exception('Predicate not implemented')),
      'SimpleSetPredicate': lambda *_: (_ for _ in ()).throw(Exception('Predicate not implemented'))
    }

    for predicate, evaluate in predicates.items():
      element = self.find(node, predicate)
      if element is not None:
        return evaluate(element, instance)

    return False


  def evaluate_simple_predicate(self, element, instance):
    field = element.get('field')

    column, mapping = self.field_mapping[field]
    a = mapping(instance[column])
    operator = element.get('operator')
    b = mapping(element.get('value'))

    operators = {
      'equal': op.eq,
      'notEqual': op.ne,
      'lessThan': op.lt,
      'lessOrEqual': op.le,
      'greaterThan': op.gt,
      'greaterOrEqual': op.ge,
      'isMissing': lambda *_: (_ for _ in ()).throw(Exception('Operator not implemented')),
      'isNotMissing': lambda *_: (_ for _ in ()).throw(Exception('Operator not implemented')),
    }

    return operators[operator](a, b)


  def predict(self, X):
    """
    Predict instances in X.

    Parameters
    ----------
    X : pd.DataFrame
        The data to be perdicted. Should have the same format as the training data.

    Returns
    -------
    numpy.ndarray
        Array of size len(X), where every row contains a prediction for the
        corresponding row in X.
    """
    if len(X.shape) != 2:
      X = X.reshape(1, 2)

    return np.array(X.apply(lambda x: self.predict_instance(x), axis=1))

  def predict_proba(self, X):
    """
    Predict instances in X.

    Parameters
    ----------
    X : pd.DataFrame
        The data to be perdicted. Should have the same format as the training data.

    Returns
    -------
    numpy.ndarray
        Array of size len(X), where every row contains a probability for each class
        for the corresponding row in X.
    """
    if len(X.shape) != 2:
      X = X.reshape(1, 2)

    return np.array(X.apply(lambda x: self.predict_instance(x, probabilities=True), axis=1))

  def predict_instance(self, instance, probabilities=False):
    """
    Perdiction for a single instance.

    Parameters
    ----------
    instance : pd.Series
        Instance to be classified.

    Returns
    -------
    Any
        Prediction values or class probabilities.

    """

    Node = self.tree

    while True:
      childNodes = self.findall(Node, 'Node')

      if len(childNodes) == 0:
        if probabilities:
          return [
            float(e.get('recordCount')) / float(Node.get('recordCount'))
            for e in self.findall(Node, "ScoreDistribution")
          ]
        else:
          return Node.get('score')


      for childNode in childNodes:
        if self.evaluate_node(childNode, instance):
          Node = childNode
          break

    return None

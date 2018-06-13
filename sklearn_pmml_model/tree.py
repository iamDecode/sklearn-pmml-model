import numpy as np
import pandas as pd
import operator as op
from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn.utils.validation import check_array
from sklearn.base import ClassifierMixin

class PMMLTreeClassifier(PMMLBaseEstimator, ClassifierMixin):
  def __init__(self, pmml):
    super(PMMLTreeClassifier, self).__init__(pmml)

    self.tree = self.find(self.root, 'TreeModel')

    if self.tree is None:
      raise Exception('PMML model does not contain TreeModel.')

    if self.target_field is not None:
      self.classes_ = np.array([
        e.get('value')
        for e in self.findall(self.target_field, 'Value')
      ])

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
    """
    Evaluate <SimplePredicate> PMML tag.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        XML Element with tag <SimplePredicate>.

    instance : pd.Series
        Instance we want to evaluate the predicate on.

    Returns
    -------
    bool
        Indicating whether the predicate holds or not.

    """
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
    X = check_array(X)

    return np.ma.apply_along_axis(lambda x: self.predict_instance(x), 1, X)

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
    X = check_array(X)

    return np.apply_along_axis(lambda x: self.predict_instance(x, probabilities=True), 1, X)

  def predict_instance(self, instance, probabilities=False):
    """
    Prediction for a single instance.

    Parameters
    ----------
    instance : pd.Series
        Instance to be classified.

    probabilities : bool (default: False)
        Whether the method should return class probabilities or just the predicted class.

    Returns
    -------
    Any
        Prediction values or class probabilities.

    """
    node = self.tree

    while True:
      childNodes = self.findall(node, 'Node')

      if len(childNodes) == 0:
        if probabilities:
          return pd.Series([
            float(e.get('recordCount')) / float(node.get('recordCount'))
            for e in self.findall(node, 'ScoreDistribution')
          ])
        else:
          return node.get('score')

      for childNode in childNodes:
        if self.evaluate_node(childNode, instance):
          node = childNode
          break

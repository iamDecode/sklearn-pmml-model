# License: BSD 2-Clause

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn_pmml_model.base import PMMLBaseClassifier, PMMLBaseRegressor, get_type
from sklearn_pmml_model.datatypes import Category
from sklearn_pmml_model.neural_network._base import PMMLBaseNeuralNetwork


class PMMLMLPClassifier(PMMLBaseClassifier, PMMLBaseNeuralNetwork, MLPClassifier):
  """
  Multi-layer Perceptron classifier.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/NeuralNetwork.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    MLPClassifier.__init__(self)
    PMMLBaseNeuralNetwork.__init__(self)

    if len(self.classes_) == 2:
      self.out_activation_ = "logistic"
      self.n_outputs_ = 1
    else:
      self.out_activation_ = "softmax"
      self.n_outputs_ = len(self.classes_)

    target_type: Category = get_type(self.target_field)
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    self._label_binarizer.classes_ = np.array(target_type.categories)
    self._label_binarizer.y_type_ = type_of_target(target_type.categories)
    self._label_binarizer.sparse_input_ = False

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return MLPClassifier._more_tags(self)


class PMMLMLPRegressor(PMMLBaseRegressor, PMMLBaseNeuralNetwork, MLPRegressor):
  """
  Multi-layer Perceptron regressor.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/NeuralNetwork.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    MLPRegressor.__init__(self)
    PMMLBaseNeuralNetwork.__init__(self)

    self.out_activation_ = "identity"

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return MLPRegressor._more_tags(self)

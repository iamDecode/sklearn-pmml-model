# License: BSD 2-Clause

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn_pmml_model.base import PMMLBaseClassifier, PMMLBaseRegressor
from sklearn_pmml_model.neighbors._base import PMMLBaseKNN


class PMMLKNeighborsClassifier(PMMLBaseClassifier, PMMLBaseKNN, KNeighborsClassifier):
  """
  Classifier implementing the k-nearest neighbors vote.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/KNN.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    KNeighborsClassifier.__init__(self)
    PMMLBaseKNN.__init__(self)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return KNeighborsClassifier._more_tags(self)


class PMMLKNeighborsRegressor(PMMLBaseRegressor, PMMLBaseKNN, KNeighborsRegressor):
  """
  Regression based on k-nearest neighbors.

  The target is predicted by local interpolation of the targets
  associated of the nearest neighbors in the training set.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/KNN.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    KNeighborsRegressor.__init__(self)
    PMMLBaseKNN.__init__(self)

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return KNeighborsRegressor._more_tags(self)

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

  n_jobs : int, default=None
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.
    Doesn't affect :meth:`fit` method.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/KNN.html

  """

  def __init__(self, pmml, n_jobs=None):
    PMMLBaseClassifier.__init__(self, pmml)
    KNeighborsClassifier.__init__(self, n_jobs=n_jobs)
    PMMLBaseKNN.__init__(self)

    KNeighborsClassifier.fit(self, self._X, self._y)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return {'requires_y': True, **KNeighborsClassifier._more_tags(self)}


class PMMLKNeighborsRegressor(PMMLBaseRegressor, PMMLBaseKNN, KNeighborsRegressor):
  """
  Regression based on k-nearest neighbors.

  The target is predicted by local interpolation of the targets
  associated of the nearest neighbors in the training set.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  n_jobs : int, default=None
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.
    Doesn't affect :meth:`fit` method.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/KNN.html

  """

  def __init__(self, pmml, n_jobs=None):
    PMMLBaseRegressor.__init__(self, pmml)
    KNeighborsRegressor.__init__(self, n_jobs=n_jobs)
    PMMLBaseKNN.__init__(self)

    KNeighborsRegressor.fit(self, self._X, self._y)

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return KNeighborsRegressor._more_tags(self)

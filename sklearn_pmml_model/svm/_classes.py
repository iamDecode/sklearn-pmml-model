# License: BSD 2-Clause

from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
import numpy as np
from scipy.sparse import isspmatrix
from sklearn_pmml_model.base import OneHotEncodingMixin, PMMLBaseClassifier, PMMLBaseRegressor
from sklearn_pmml_model.svm._base import PMMLBaseSVM
from sklearn_pmml_model.linear_model.implementations import _get_coefficients as _linear_get_coefficients


class PMMLLinearSVC(OneHotEncodingMixin, PMMLBaseClassifier, LinearSVC):
  """
  Linear Support Vector Classification.

  Similar to SVC with parameter kernel='linear', but implemented in terms of
  liblinear rather than libsvm, so it has more flexibility in the choice of
  penalties and loss functions and should scale better to large numbers of
  samples.

  This class supports both dense and sparse input and the multiclass support
  is handled according to a one-vs-the-rest scheme.

  The PMML model is assumed to be equivalent to PMMLLogisticRegression.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/Regression.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)
    LinearSVC.__init__(self)

    # Import coefficients and intercepts
    model = self.root.find('RegressionModel')

    if model is None:
      raise Exception('PMML model does not contain RegressionModel.')

    tables = [
      table for table in model.findall('RegressionTable')
      if table.find('NumericPredictor') is not None
    ]

    self.coef_ = [
      _linear_get_coefficients(self, table)
      for table in tables
    ]
    self.intercept_ = [
      float(table.get('intercept'))
      for table in tables
    ]

    if len(self.coef_) == 1:
      self.coef_ = [self.coef_[0]]

    if len(self.intercept_) == 1:
      self.intercept_ = [self.intercept_[0]]

    self.coef_ = np.array(self.coef_)
    self.intercept_ = np.array(self.intercept_)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return LinearSVC._more_tags(self)


class PMMLLinearSVR(OneHotEncodingMixin, PMMLBaseRegressor, LinearSVR):
  """
  Linear Support Vector Regression.

  Similar to SVR with parameter kernel='linear', but implemented in terms of
  liblinear rather than libsvm, so it has more flexibility in the choice of
  penalties and loss functions and should scale better to large numbers of
  samples.

  This class supports both dense and sparse input.

  The PMML model is assumed to be equivalent to PMMLLinearRegression.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/Regression.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)

    # Import coefficients and intercepts
    model = self.root.find('RegressionModel')

    if model is None:
      raise Exception('PMML model does not contain RegressionModel.')

    tables = model.findall('RegressionTable')

    self.coef_ = np.array([
      _linear_get_coefficients(self, table)
      for table in tables
    ])
    self.intercept_ = np.array([
      float(table.get('intercept'))
      for table in tables
    ])

    if self.coef_.shape[0] == 1:
      self.coef_ = self.coef_[0]

    if self.intercept_.shape[0] == 1:
      self.intercept_ = self.intercept_[0]

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return LinearSVR._more_tags(self)


class PMMLNuSVC(OneHotEncodingMixin, PMMLBaseClassifier, PMMLBaseSVM, NuSVC):
  """
  Nu-Support Vector Classification.

  Similar to SVC but uses a parameter to control the number of support
  vectors.

  The implementation is based on libsvm.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/SupportVectorMachine.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)
    NuSVC.__init__(self)
    PMMLBaseSVM.__init__(self)

  def _prepare_data(self, X):
    self._sparse = isspmatrix(X)
    return super()._prepare_data(X)

  def decision_function(self, X, *args, **kwargs):
    X = self._prepare_data(X)
    return super().decision_function(X, *args, **kwargs)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return NuSVC._more_tags(self)


class PMMLNuSVR(OneHotEncodingMixin, PMMLBaseRegressor, PMMLBaseSVM, NuSVR):
  """
  Nu Support Vector Regression.

  Similar to NuSVC, for regression, uses a parameter nu to control
  the number of support vectors. However, unlike NuSVC, where nu
  replaces C, here nu replaces the parameter epsilon of epsilon-SVR.

  The implementation is based on libsvm.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/SupportVectorMachine.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)
    NuSVR.__init__(self)
    PMMLBaseSVM.__init__(self)

  def _prepare_data(self, X):
    self._sparse = isspmatrix(X)
    return super()._prepare_data(X)

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return NuSVR._more_tags(self)


class PMMLSVC(OneHotEncodingMixin, PMMLBaseClassifier, PMMLBaseSVM, SVC):
  """
  C-Support Vector Classification.

  The implementation is based on libsvm. The multiclass support is
  handled according to a one-vs-one scheme.

  For details on the precise mathematical formulation of the provided
  kernel functions and how `gamma`, `coef0` and `degree` affect each
  other, see the corresponding section in the narrative documentation:
  `Kernel functions <https://scikit-learn.org/stable/modules/svm.html#svm-kernels>`_.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/SupportVectorMachine.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)
    SVC.__init__(self)
    PMMLBaseSVM.__init__(self)

  def _prepare_data(self, X):
    self._sparse = isspmatrix(X)
    return super()._prepare_data(X)

  def decision_function(self, X, *args, **kwargs):
    X = self._prepare_data(X)
    return super().decision_function(X, *args, **kwargs)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return SVC._more_tags(self)


class PMMLSVR(OneHotEncodingMixin, PMMLBaseRegressor, PMMLBaseSVM, SVR):
  """
  Epsilon-Support Vector Regression.

  The free parameters in the model are C and epsilon. The implementation
  is based on libsvm.

  For details on the precise mathematical formulation of the provided
  kernel functions and how `gamma`, `coef0` and `degree` affect each
  other, see the corresponding section in the narrative documentation:
  `Kernel functions <https://scikit-learn.org/stable/modules/svm.html#svm-kernels>`_.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/SupportVectorMachine.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)
    SVR.__init__(self)
    PMMLBaseSVM.__init__(self)

  def _prepare_data(self, X):
    self._sparse = isspmatrix(X)
    return super()._prepare_data(X)

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return SVR._more_tags(self)

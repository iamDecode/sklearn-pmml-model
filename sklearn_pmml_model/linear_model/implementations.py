# License: BSD 2-Clause

from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, LogisticRegression
from sklearn_pmml_model.base import PMMLBaseRegressor, PMMLBaseClassifier, OneHotEncodingMixin
from sklearn_pmml_model.linear_model.base import PMMLGeneralizedLinearRegressor, PMMLGeneralizedLinearClassifier
from itertools import chain
import numpy as np


class PMMLLinearRegression(OneHotEncodingMixin, PMMLBaseRegressor, LinearRegression):
  """
  Ordinary least squares Linear Regression.

  The PMML model consists out of a <RegressionModel> element, containing at
  least one <RegressionTable> element. Every table element contains a
  <NumericPredictor> element for numerical fields and <CategoricalPredictor>
  per value of a categorical field, describing the coefficients.

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
      _get_coefficients(self, table)
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
    return LinearRegression._more_tags(self)


class PMMLLogisticRegression(OneHotEncodingMixin, PMMLBaseClassifier, LogisticRegression):
  """
  Logistic Regression (aka logit, MaxEnt) classifier.

  The PMML model consists out of a <RegressionModel> element, containing at
  least one <RegressionTable> element. Every table element contains a
  <NumericPredictor> element for numerical fields and <CategoricalPredictor>
  per value of a categorical field, describing the coefficients.

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

    # Import coefficients and intercepts
    model = self.root.find('RegressionModel')
    mining_model = self.root.find('MiningModel')
    tables = []

    if mining_model is not None and self.n_classes_ > 2:
      self.multi_class = 'ovr'
      segmentation = mining_model.find('Segmentation')

      if segmentation.get('multipleModelMethod') not in ['modelChain']:
        raise Exception('PMML model for multi-class logistic regression should use modelChain method.')

      # Parse segments
      segments = segmentation.findall('Segment')
      valid_segments = [segment for segment in segments if segment.find('True') is not None]
      models = [segment.find('RegressionModel') for segment in valid_segments]

      tables = [
        models[i].find('RegressionTable') for i in range(self.n_classes_)
      ]
    elif model is not None:
      self.multi_class = 'auto'
      tables = [
        table for table in model.findall('RegressionTable')
        if table.find('NumericPredictor') is not None
      ]
    else:
      raise Exception('PMML model does not contain RegressionModel or Segmentation.')

    self.coef_ = [
      _get_coefficients(self, table)
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
    self.solver = 'lbfgs'

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return LogisticRegression._more_tags(self)


def _get_coefficients(est, table):
  """
  Obtain coefficients for <RegressionTable> PMML elements.

  Parameters
  ----------
  est : PMMLBaseEstimator
    Base estimator containing information about `fields` and `field_mapping`.

  table : eTree.Element
      The <RegressionTable> element which contains the feature coefficients.

  """
  def coefficient_for_category(predictors, category):
    predictor = [p for p in predictors if p.get('value') == category]

    if not predictor:
      return 0

    return float(predictor[0].get('coefficient'))

  def coefficients_for_field(name, field):
    predictors = table.findall(f"*[@name='{name}']")

    if field.get('optype') != 'categorical':
      if len(predictors) > 1:
        raise Exception('PMML model is not linear.')

      return [float(predictors[0].get('coefficient'))]

    return [
      coefficient_for_category(predictors, c)
      for c in est.field_mapping[name][1].categories
    ]

  return list(chain.from_iterable([
    coefficients_for_field(name, field)
    for name, field in est.fields.items()
    if table.find(f"*[@name='{name}']") is not None
  ]))


# NOTE: Many of these variants only differ in the training part, not the
# classification part. Hence they are equivalent in terms of parsing.


class PMMLRidge(PMMLGeneralizedLinearRegressor, Ridge):
  """
  Linear least squares with l2 regularization.

  Minimizes the objective function::

  ||y - Xw||^2_2 + alpha * ||w||^2_2

  This model solves a regression model where the loss function is
  the linear least squares function and regularization is given by
  the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
  This estimator has built-in support for multi-variate regression
  (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return Ridge._more_tags(self)


class PMMLRidgeClassifier(PMMLGeneralizedLinearClassifier, RidgeClassifier):
  """
  Classifier using Ridge regression.

  This classifier first converts the target values into ``{-1, 1}`` and
  then treats the problem as a regression task (multi-output regression in
  the multiclass case).

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def __init__(self, pmml):
    PMMLGeneralizedLinearClassifier.__init__(self, pmml)
    RidgeClassifier.__init__(self)

  def fit(self, x, y):
    return PMMLGeneralizedLinearClassifier.fit(self, x, y)

  def _more_tags(self):
    return RidgeClassifier._more_tags(self)


class PMMLLasso(PMMLGeneralizedLinearRegressor, Lasso):
  """
  Linear Model trained with L1 prior as regularizer (aka the Lasso).

  The optimization objective for Lasso is::

      (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

  Technically the Lasso model is optimizing the same objective function as
  the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def __init__(self, pmml):
    PMMLGeneralizedLinearRegressor.__init__(self, pmml)
    self.n_iter_ = 0

  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return Lasso._more_tags(self)


class PMMLElasticNet(PMMLGeneralizedLinearRegressor, ElasticNet):
  """
  Linear regression with combined L1 and L2 priors as regularizer.

  Minimizes the objective function::

          1 / (2 * n_samples) * ||y - Xw||^2_2
          + alpha * l1_ratio * ||w||_1
          + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

  If you are interested in controlling the L1 and L2 penalty
  separately, keep in mind that this is equivalent to::

          a * ||w||_1 + 0.5 * b * ||w||_2^2

  where::

          alpha = a + b and l1_ratio = a / (a + b)

  The parameter l1_ratio corresponds to alpha in the glmnet R package while
  alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
  = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
  unless you supply your own sequence of alpha.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def __init__(self, pmml):
    PMMLGeneralizedLinearRegressor.__init__(self, pmml)
    self.n_iter_ = 0

  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return ElasticNet._more_tags(self)

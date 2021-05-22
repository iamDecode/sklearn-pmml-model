from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, LogisticRegression
from sklearn_pmml_model.linear_model.base import PMMLLinearModel, PMMLLinearClassifier, PMMLGeneralizedLinearRegressor,\
  PMMLGeneralizedLinearClassifier
from itertools import chain
import numpy as np


class PMMLLinearRegression(PMMLLinearModel, LinearRegression):
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
    PMMLLinearModel.__init__(self, pmml)

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
    return PMMLLinearModel.fit(self, x, y)

  def _more_tags(self):
    return LinearRegression._more_tags(self)


class PMMLLogisticRegression(PMMLLinearClassifier, LogisticRegression):
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
    PMMLLinearClassifier.__init__(self, pmml)

    # Import coefficients and intercepts
    model = self.root.find('RegressionModel')

    if model is None:
      raise Exception('PMML model does not contain RegressionModel.')

    tables = [
      table for table in model.findall('RegressionTable')
      if table.find('NumericPredictor') is not None
    ]

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
    self.multi_class = 'auto'

  def fit(self, x, y):
    return PMMLLinearClassifier.fit(self, x, y)

  def _more_tags(self):
    return LogisticRegression._more_tags(self)


def _get_coefficients(est, table):
  """
  Helper method to obtain coefficients for <RegressionTable> PMML elements.

  Parameters
  ----------
  est : PMMLBaseEstimator
    Base estimator containing information about `fields` and `field_mapping`.

  table: eTree.Element
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


'''
NOTE: Many of these variants only differ in the training part, not the 
classification part. Hence they are equavalent in terms of parsing.
'''


class PMMLRidge(PMMLGeneralizedLinearRegressor, Ridge):
  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return Ridge._more_tags(self)


class PMMLRidgeClassifier(PMMLGeneralizedLinearClassifier, RidgeClassifier):
  def __init__(self, pmml):
    PMMLGeneralizedLinearClassifier.__init__(self, pmml)
    RidgeClassifier.__init__(self)

  def fit(self, x, y):
    return PMMLGeneralizedLinearClassifier.fit(self, x, y)

  def _more_tags(self):
    return RidgeClassifier._more_tags(self)


class PMMLLasso(PMMLGeneralizedLinearRegressor, Lasso):
  def __init__(self, pmml):
    PMMLGeneralizedLinearRegressor.__init__(self, pmml)
    self.n_iter_ = 0

  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return Lasso._more_tags(self)


class PMMLElasticNet(PMMLGeneralizedLinearRegressor, ElasticNet):
  def __init__(self, pmml):
    PMMLGeneralizedLinearRegressor.__init__(self, pmml)
    self.n_iter_ = 0

  def fit(self, x, y):
    return PMMLGeneralizedLinearRegressor.fit(self, x, y)

  def _more_tags(self):
    return ElasticNet._more_tags(self)
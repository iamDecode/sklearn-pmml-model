# License: BSD 2-Clause

from sklearn_pmml_model.base import PMMLBaseRegressor, PMMLBaseClassifier, OneHotEncodingMixin
import numpy as np
from itertools import chain


class PMMLGeneralizedLinearRegressor(OneHotEncodingMixin, PMMLBaseRegressor):
  """
  Abstract class for Generalized Linear Models (GLMs).

  The PMML model consists out of a <GeneralRegressionModel> element,
  containing a <ParamMatrix> element that contains zero or more <PCell>
  elements describing the coefficients for each parameter. Parameters
  are described in the <PPMatrix> element, that maps parameters to fields in
  the data.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def __init__(self, pmml):
    PMMLBaseRegressor.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)

    # Import coefficients and intercepts
    model = self.root.find('GeneralRegressionModel')

    if model is None:
      raise Exception('PMML model does not contain GeneralRegressionModel.')

    self.coef_ = np.array(_get_coefficients(self, model))
    self.intercept_ = _get_intercept(model)


class PMMLGeneralizedLinearClassifier(OneHotEncodingMixin, PMMLBaseClassifier):
  """
  Abstract class for Generalized Linear Models (GLMs).

  The PMML model consists out of a <GeneralRegressionModel> element,
  containing a <ParamMatrix> element that contains zero or more <PCell>
  elements describing the coefficients for each parameter. Parameters
  are described in the <PPMatrix> element, that maps parameters to fields in
  the data.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/GeneralRegression.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)

    # Import coefficients and intercepts
    model = self.root.find('GeneralRegressionModel')

    if model is None:
      raise Exception('PMML model does not contain GeneralRegressionModel.')

    self.coef_ = np.array([_get_coefficients(self, model)])
    self.intercept_ = _get_intercept(model)


def _get_coefficients(linear_model, model):
  """
  Obtain the coefficients for the GLM regression.

  Raises an exception when we notice non linear parameter configurations.

  Parameters
  ----------
  linear_model : PMMLGeneralizedLinearRegressor, PMMLGeneralizedLinearClassifier
      The PMML class representing the classifier. Should contain at least target_field,
      fields and field_mapping properties.

  model : eTree.Element
      The <GeneralRegressionModel> element that is assumed to contains a
      <PPMatrix> and <ParamMatrix> element.

  Returns
  -------
  coefficients: numpy.ndarray
      Coefficient value for every field. Zero if not present.

  """
  pp = model.find('PPMatrix')
  params = model.find('ParamMatrix')

  def coefficient_for_parameter(p):
    if not p:
      return 0

    pcells = params.findall(f"PCell[@parameterName='{p}']")
    if len(pcells) > 1:
      raise Exception('This model does not support multiple outputs.')

    if not pcells:
      return 0

    return float(pcells[0].get('beta'))

  def parameter_for_category(cells, category):
    cell = [cell for cell in cells if cell.get('value') == category]

    if not cell:
      return None

    return cell[0].get('parameterName')

  def coefficients_for_field(name, field):
    pp_cells = pp.findall(f"PPCell[@predictorName='{name}']")

    if not pp_cells:
      return [0]

    if field.get('optype') != 'categorical':
      if len(pp_cells) > 1:
        raise Exception('PMML model is not linear.')

      return [coefficient_for_parameter(pp_cells[0].get('parameterName'))]

    return [
      coefficient_for_parameter(parameter_for_category(pp_cells, c))
      for c in linear_model.field_mapping[name][1].categories
    ]

  target = linear_model.target_field.get('name')
  fields = {name: field for name, field in linear_model.fields.items() if name != target}

  return list(chain.from_iterable([
    coefficients_for_field(name, field)
    for name, field in fields.items()

  ]))


def _get_intercept(model):
  """
  Find all parameters that are not included in the <ParamMatrix>.

  These constitute the intercept. In the very unlikely case there are multiple
  parameters fitting this criteria, we sum the result.

  Parameters
  ----------
  model : eTree.Element
      The <GeneralRegressionModel> element that is assumed to contains a
      <PPMatrix> and <ParamMatrix> element.

  Returns
  -------
  intercept : float
      Value of the intercept of the method.

  """
  pp = model.find('PPMatrix')
  params = model.find('ParamMatrix')

  specified = [p.get('parameterName') for p in pp.findall('PPCell')]
  used = [p.get('parameterName') for p in params.findall('PCell')]

  intercepts = set(used) - set(specified)
  intercepts = list(chain.from_iterable([
    params.findall(f"PCell[@parameterName='{p}']")
    for p in intercepts
  ]))

  return sum([float(i.get('beta')) for i in intercepts])

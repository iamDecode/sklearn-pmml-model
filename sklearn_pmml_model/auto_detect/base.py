from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn_pmml_model.datatypes import Category
from sklearn_pmml_model.tree import PMMLTreeClassifier, PMMLTreeRegressor
from sklearn_pmml_model.ensemble import PMMLForestClassifier, PMMLForestRegressor, PMMLGradientBoostingClassifier, \
  PMMLGradientBoostingRegressor
from sklearn_pmml_model.neural_network import PMMLMLPClassifier, PMMLMLPRegressor
from sklearn_pmml_model.svm import PMMLSVC, PMMLSVR
from sklearn_pmml_model.naive_bayes import PMMLGaussianNB
from sklearn_pmml_model.linear_model import PMMLLogisticRegression, PMMLLinearRegression, PMMLRidgeClassifier, PMMLRidge
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier, PMMLKNeighborsRegressor


def auto_detect_estimator(pmml, **kwargs):
  """
  Automatically detect and return the described estimator from PMML file.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  base = PMMLBaseEstimator(pmml=pmml)
  target_field_name = base.target_field.attrib['name']
  target_field_type = base.field_mapping[target_field_name][1]

  if isinstance(target_field_type, Category) or target_field_type is str:
    return auto_detect_classifier(pmml, **kwargs)
  else:
    return auto_detect_regressor(pmml, **kwargs)


def auto_detect_classifier(pmml, **kwargs):
  """
  Automatically detect and return the described classifier from PMML file.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  if isinstance(pmml, str):
    file = open(pmml, 'r')
  else:
    pmml.seek(0)
    file = pmml

  for line in file:
    if '<Segmentation' in line:
      clfs = [x for x in (detect_classifier(line) for line in file) if x is not None]
      file.close()

      if all(clf is PMMLTreeClassifier or clf is PMMLLogisticRegression for clf in clfs):
        if 'multipleModelMethod="majorityVote"' in line or 'multipleModelMethod="average"' in line:
          return PMMLForestClassifier(pmml=pmml, **kwargs)
        if 'multipleModelMethod="modelChain"' in line:
          return PMMLGradientBoostingClassifier(pmml=pmml, **kwargs)

      raise Exception('Unsupported PMML classifier: invalid segmentation.')

    clf = detect_classifier(line)
    if clf:
      file.close()
      return clf(pmml, **kwargs)

  file.close()
  raise Exception('Unsupported PMML classifier.')


def auto_detect_regressor(pmml, **kwargs):
  """
  Automatically detect and return the described regressor from PMML file.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  if isinstance(pmml, str):
    file = open(pmml, 'r')
  else:
    pmml.seek(0)
    file = pmml

  for line in file:
    if '<Segmentation' in line:
      regs = [x for x in (detect_regressor(line) for line in file) if x is not None]
      file.close()

      if all(reg is PMMLTreeRegressor or reg is PMMLLinearRegression for reg in regs):
        if 'multipleModelMethod="majorityVote"' in line or 'multipleModelMethod="average"' in line:
          return PMMLForestRegressor(pmml=pmml, **kwargs)
        if 'multipleModelMethod="sum"' in line:
          return PMMLGradientBoostingRegressor(pmml=pmml, **kwargs)

      raise Exception('Unsupported PMML regressor: invalid segmentation.')

    reg = detect_regressor(line)
    if reg:
      file.close()
      return reg(pmml, **kwargs)

  file.close()
  raise Exception('Unsupported PMML regressor.')


def detect_classifier(line):
  """
  Detect the type of classifier in line if present.

  Parameters
  ----------
  line : str
      Line of a PMML file as a string.

  pmml : str, object
      Filename or file object containing PMML data.

  """
  if '<TreeModel' in line:
    return PMMLTreeClassifier

  if '<NeuralNetwork' in line:
    return PMMLMLPClassifier

  if '<SupportVectorMachineModel' in line:
    return PMMLSVC

  if '<NaiveBayesModel' in line:
    return PMMLGaussianNB

  if '<GeneralRegressionModel' in line:
    return PMMLRidgeClassifier

  if '<RegressionModel' in line:
    return PMMLLogisticRegression

  if '<NearestNeighborModel' in line:
    return PMMLKNeighborsClassifier

  return None


def detect_regressor(line):
  """
  Detect the type of regressor in line if present.

  Parameters
  ----------
  line : str
      Line of a PMML file as a string.

  pmml : str, object
      Filename or file object containing PMML data.

  """
  if '<TreeModel' in line:
    return PMMLTreeRegressor

  if '<NeuralNetwork' in line:
    return PMMLMLPRegressor

  if '<SupportVectorMachineModel' in line:
    return PMMLSVR

  if '<GeneralRegressionModel' in line:
    return PMMLRidge

  if '<RegressionModel' in line:
    return PMMLLinearRegression

  if '<NearestNeighborModel' in line:
    return PMMLKNeighborsRegressor

  return None

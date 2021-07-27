# License: BSD 2-Clause

from sklearn_pmml_model.base import PMMLBaseRegressor, parse_array
import numpy as np


class PMMLBaseSVM:
  """
  Abstract class for Support Vector Machines.

  The PMML model consists out of a <SupportVectorMachineModel> element,
  containing a <SupportVectorMachine> element that contains a <SupportVectors>
  element describing support vectors, and a <Coefficients> element describing
  the coefficients for each support vector. Support vectors are referenced from
  a <VectorDictionary> element, in which the true support vectors are described
  using <VectorInstance> elements. Furthermore, the model contains one out of
  <LinearKernelType>, <PolynomialKernelType>, <RadialBasisKernelType> or
  <SigmoidKernelType> describing the kernel function used.

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/SupportVectorMachineModel.html

  """

  def __init__(self):
    # Import coefficients and intercepts
    model = self.root.find('SupportVectorMachineModel')

    if model is None:
      raise Exception('PMML model does not contain SupportVectorMachineModel.')

    vector_dictionary = model.find('VectorDictionary')
    svms = model.findall('SupportVectorMachine')
    coefficients = [svm.find('Coefficients') for svm in svms]

    self.shape_fit_ = (0, len(vector_dictionary.find('VectorFields')))
    self.support_ = np.array([
      int(x.get('id'))
      for x in vector_dictionary.findall('VectorInstance')
    ]).astype(np.int32)

    classes = [None, None] if isinstance(self, PMMLBaseRegressor) else self.classes_

    self._n_support = np.array([
      len(get_overlapping_vectors(get_alt_svms(svms, classes, c)))
      for c in classes
    ]).astype(np.int32)

    self.support_vectors_ = np.array([
      get_vectors(vector_dictionary, s) for s in self.support_
    ])

    self._intercept_ = self.intercept_ = np.array([float(cs.get('absoluteValue')) for cs in coefficients])
    self._dual_coef_ = self.dual_coef_ = np.array(
      get_coefficients(classes, self._n_support, self.support_, svms)
    )

    if len(classes) == 2:
      self._n_support = (self._n_support / 2).astype(np.int32)

    linear = model.find('LinearKernelType')
    poly = model.find('PolynomialKernelType')
    rbf = model.find('RadialBasisKernelType')
    sigmoid = model.find('SigmoidKernelType')

    if linear is not None:
      self.kernel = 'linear'
      self._gamma = self.gamma = 0.0
    elif poly is not None:
      self.kernel = 'poly'
      self._gamma = self.gamma = float(poly.get('gamma'))
      self.coef0 = float(poly.get('coef0'))
      self.degree = int(poly.get('degree'))
    elif rbf is not None:
      self.kernel = 'rbf'
      self._gamma = self.gamma = float(rbf.get('gamma'))
    elif sigmoid is not None:
      self.kernel = 'sigmoid'
      self._gamma = self.gamma = float(sigmoid.get('gamma'))
      self.coef0 = float(sigmoid.get('coef0'))

    self._probA = np.array([])
    self._probB = np.array([])


def get_vectors(vector_dictionary, s):
  """Return support vector values, parsed as a numpy array."""
  instance = vector_dictionary.find(f"VectorInstance[@id='{s}']")

  if instance is None:
    raise Exception(f'PMML model is broken, vector instance (id = {s}) not found.')

  array = instance.find('Array')
  if array is None:
    array = instance.find('REAL-Array')
  if array is None:
    array = instance.find('SparseArray')
  if array is None:
    array = instance.find('REAL-SparseArray')
  if array is None:
    raise Exception(f'PMML model is broken, vector instance (id = {s}) does not contain (Sparse)Array element.')

  return np.array(parse_array(array))


def get_alt_svms(svms, classes, target_class):
  """
  Find alternative SVMs (e.g., for target class 0, find the svms classifying 0 against 1, and 0 against 2).

  Parameters
  ----------
  svms : list
      List of eTree.Element objects describing the different one-to-one support vector machines in the PMML.

  classes : numpy.array
      The classes to be predicted by the model.

  target_class : str
      The target class.

  Returns
  -------
  alt_svms : list
      List of eTree.Elements filtered to only include SVMs comparing the target class against alternate classes.

  """
  # Noop for regression
  if classes[0] is None:
    return svms

  alt_svms = [
    svm for svm in svms
    if svm.get('targetCategory') == str(target_class) or svm.get('alternateTargetCategory') == str(target_class)
  ]

  # Sort svms based on target class order
  alt_svms = [
    next(svm for svm in alt_svms if svm.get('targetCategory') == str(c) or svm.get('alternateTargetCategory') == str(c))
    for c in set(classes).difference({target_class})
  ]

  return alt_svms


def get_overlapping_vectors(svms):
  """
  Return support vector ids that are present in all provided SVM elements.

  Parameters
  ----------
  svms : list
      List of eTree.Element objects describing the different one-to-one support vector machines in the PMML.

  Returns
  -------
  output : set
    Set containing all integer vector ids that are present in all provided SVM elements.

  """
  support_vectors = [svm.find('SupportVectors') for svm in svms]
  vector_ids = [{int(x.get('vectorId')) for x in s.findall('SupportVector')} for s in support_vectors]
  return set.intersection(*vector_ids)


def get_coefficients(classes, n_support, support_ids, svms):
  """
  Return support vector coefficients.

  Parameters
  ----------
  classes : numpy.array
      The classes to be predicted by the model.

  n_support : numpy.array
      Numpy array describing the number of support vectors for each class.

  support_ids: list
    A list describing the ids of all support vectors in the model.

  svms : list
      List of eTree.Element objects describing the different one-to-one support vector machines in the PMML.

  """
  dual_coef = np.zeros((len(classes) - 1, len(support_ids)))

  for i, x in enumerate(classes):
    alt_svms = get_alt_svms(svms, classes, x)
    offsets = [0] + np.cumsum(n_support).tolist()

    for j, svm in enumerate(alt_svms):
      start = offsets[i]
      end = offsets[i + 1]
      ids = support_ids[start:end]

      support_vectors = [int(x.get('vectorId')) for x in svm.find('SupportVectors').findall('SupportVector')]
      coefficients = [float(x.get('value')) for x in svm.find('Coefficients').findall('Coefficient')]
      indices = [support_vectors.index(x) for x in ids]
      dual_coef[j, start:end] = np.array(coefficients)[indices]

  return dual_coef

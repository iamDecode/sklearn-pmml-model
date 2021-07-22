from sklearn_pmml_model.base import parse_array
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
    svm = model.find('SupportVectorMachine')
    support_vectors = svm.find('SupportVectors')
    coefficients = svm.find('Coefficients')

    self.shape_fit_ = (0, len(vector_dictionary.find('VectorFields')))
    self._intercept_ = self.intercept_ = np.array([float(coefficients.get('absoluteValue'))])
    self._dual_coef_ = self.dual_coef_ = np.array([[
      float(c.get('value'))
      for c in coefficients.findall("Coefficient")
    ]])
    self.support_ = np.array([
      int(s.get('vectorId'))
      for s in support_vectors.findall('SupportVector')
    ]).astype(np.int32)
    self._n_support = (np.repeat(len(self.support_), self.n_classes_) / self.n_classes_).astype(np.int32)
    self.support_vectors_ = np.array([get_vectors(vector_dictionary, s) for s in self.support_])

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
    else:
      raise Exception('Unknown or missing kernel type.')

    self._probA = np.array([])
    self._probB = np.array([])


def get_vectors(vector_dictionary, s):
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

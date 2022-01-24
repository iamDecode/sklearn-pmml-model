# License: BSD 2-Clause

import numpy as np
import pandas as pd


class PMMLBaseKNN:
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

  def __init__(self, leaf_size=30):
    knn_model = self.root.find('NearestNeighborModel')

    if knn_model is None:
      raise Exception('PMML model does not contain NearestNeighborModel.')

    self.n_neighbors = int(knn_model.get('numberOfNeighbors'))
    self.algorithm = 'auto'
    self.leaf_size = leaf_size
    self.p = 2
    self.metric_params = None
    self.outputs_2d_ = False

    # Set metric and parameters
    measure_element = knn_model.find('ComparisonMeasure')

    if measure_element is None:
      raise Exception('PMML model does not contain ComparisonMeasure.')

    measure = next(x for x in measure_element)

    measures = {
      'euclidean': 'euclidean',
      'chebychev': 'chebyshev',
      'cityBlock': 'manhattan',
      'minkowski': 'minkowski',
      'simpleMatching': 'matching',
      'jaccard': 'jaccard',
      'tanimoto': 'rogerstanimoto',
    }

    if measure.tag not in measures:
      raise Exception(f'PMML model uses unsupported distance metric: "{measure.tag}".')

    self.metric = measures[measure.tag]

    if self.metric == 'minkowski':
      self.p = float(measure.get('p-parameter'))
      self.metric_params = {'p': self.p}

    self._check_algorithm_metric()

    # Set training instances
    instances = knn_model.find('TrainingInstances')

    fields_element = instances.find('InstanceFields')
    mapping = {x.get('field'): x.get('column').split(':')[-1] for x in fields_element}
    target = self.target_field.get('name')
    fields = [x.get('field') for x in fields_element if x.get('field') != target]

    data = [
      [
        self.field_mapping[f][1](next(x for x in row if x.tag.endswith(mapping[f])).text)
        for f in fields
      ]
      for row in instances.find('InlineTable')
    ]

    self._X = pd.DataFrame(data, columns=fields)
    self._y = np.array([
      self.field_mapping[target][1](next(x for x in row if x.tag.endswith(mapping[target])).text)
      for row in instances.find('InlineTable')
    ])

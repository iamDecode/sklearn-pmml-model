# License: BSD 2-Clause

import numpy as np
from sklearn_pmml_model.base import PMMLBaseClassifier


class PMMLBaseNeuralNetwork:
  """
  Abstract class for Neural Network models.

  The PMML model consists out of a <NeuralNetwork> element, containing a
  <NeuralInputs> element that describes the input layer neurons with
  <NeuralInput> elements. Next, a <NeuralLayer> element describes all other
  neurons with associated weights and biases. The activation function is either
  specified globally with the activationFunction attribute on the
  <NeuralNetwork> element, or the same attribute on each layer. Note however
  that scikit-learn only supports a single activation function for all hidden
  layers. Finally, the <NeuralOutputs> element describes the output layer.
  The output is currently expected to match the target field in <MiningSchema>.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/NeuralNetwork.html

  """

  def __init__(self):
    nn_model = self.root.find('NeuralNetwork')

    if nn_model is None:
      raise Exception('PMML model does not contain NeuralNetwork.')

    inputs = nn_model.find('NeuralInputs')

    if inputs is None:
      raise Exception('PMML model does not contain NeuralInputs.')

    mapping = {
      x.find('DerivedField').find('FieldRef').get('field'): x.get('id')
      for x in inputs.findall('NeuralInput')
    }

    target = self.target_field.get('name')
    fields = [name for name, field in self.fields.items() if name != target and field.tag == 'DataField']
    if set(mapping.keys()) != set(fields):
      raise Exception('PMML model preprocesses the data which currently unsupported.')

    layers = [layer for layer in nn_model.findall('NeuralLayer')]
    if isinstance(self, PMMLBaseClassifier) and len(self.classes_) == 2:
      index = next((i + 1 for i, layer in enumerate(layers) if layer.get('activationFunction') == 'identity'), None)
      layers = layers[:index]

    if len(layers) == 0:
      raise Exception('PMML model does not contain any NeuralLayer elements.')

    self.n_layers_ = len(layers) + 1  # +1 for input layer

    neurons = [layer.findall('Neuron') for layer in layers]
    self.hidden_layer_sizes = [len(neuron) for neuron in neurons][:-1]

    # Determine activation function
    activation_functions = {
      'logistic': 'logistic',
      'tanh': 'tanh',
      'identity': 'identity',
      'rectifier': 'relu'
    }
    activation_function = nn_model.get('activationFunction')

    if activation_function is None:
      activation_function = layers[0].get('activationFunction')

    layer_activations = [
      layer.get('activationFunction')
      for layer in layers[:-1]
      if layer.get('activationFunction') is not None
    ]

    if len(np.unique([activation_function] + layer_activations)) > 1:
      raise Exception('Neural networks with different activation functions per '
                      'layer are not currently supported by scikit-learn.')

    if activation_function not in activation_functions:
      raise Exception('PMML model uses unsupported activationFunction.')

    self.activation = activation_functions[activation_function]

    # Set neuron weights
    sizes = list(zip(
      [len(mapping)] + [len(layer) for layer in layers][:-1],
      [len(layer) for layer in layers]
    ))

    self.coefs_ = [np.zeros(shape=s) for s in sizes]
    self.intercepts_ = [
      np.array([float(neuron.get('bias', 0)) for neuron in layer])
      for layer in neurons
    ]

    field_ids = [mapping[field] for field in fields]
    for li, layer in enumerate(neurons):
      if li == 0:
        layer_ids = field_ids
      else:
        layer_ids = [x.get('id') for x in neurons[li - 1]]
      for ni, neuron in enumerate(layer):
        for connection in neuron.findall('Con'):
          ci = layer_ids.index(connection.get('from'))
          self.coefs_[li][ci, ni] = float(connection.get('weight'))

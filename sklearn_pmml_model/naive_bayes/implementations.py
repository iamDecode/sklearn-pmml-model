# License: BSD 2-Clause

from sklearn_pmml_model.base import PMMLBaseClassifier, OneHotEncodingMixin
from sklearn.naive_bayes import GaussianNB
import numpy as np
from itertools import chain


class PMMLGaussianNB(OneHotEncodingMixin, PMMLBaseClassifier, GaussianNB):
  """
  Gaussian Naive Bayes classifier.

  Can perform online updates to model parameters via :meth:`partial_fit`.
  For details on algorithm used to update feature means and variance online,
  see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

      http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

  Parameters
  ----------
  pmml : str, object
    Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/NaiveBayes.html

  """

  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)
    OneHotEncodingMixin.__init__(self)

    model = self.root.find('NaiveBayesModel')

    if model is None:
      raise Exception('PMML model does not contain NaiveBayesModel.')

    inputs = model.find('BayesInputs')

    target_values = {
      target: self._get_target_values(inputs, target)
      for target in self.classes_
    }

    try:
      outputs = model.find('BayesOutput').find('TargetValueCounts').findall('TargetValueCount')
      counts = [int(x.get('count')) for x in outputs]
      self.class_prior_ = np.array([x / np.sum(counts) for x in counts])
    except AttributeError:
      self.class_prior_ = np.array([1 / len(self.classes_) for _ in self.classes_])

    self.theta_ = np.array([
      [float(value.get('mean', 0)) for value in target_values[target]]
      for target in self.classes_
    ])
    try:
      self.sigma_ = np.array([
        [float(value.get('variance', 0)) for value in target_values[target]]
        for target in self.classes_
      ])
    except AttributeError:
      self.var_ = np.array([
        [float(value.get('variance', 0)) for value in target_values[target]]
        for target in self.classes_
      ])

  def _get_target_values(self, inputs, target):
    def target_value_for_category(bayesInput, category):
      counts = bayesInput.find(f"PairCounts[@value='{category}']")
      target_counts = counts.find('TargetValueCounts')
      return target_counts.find(f"TargetValueCount[@value='{target}']")

    def target_value_for_field(name, field):
      bayesInput = inputs.find(f"BayesInput[@fieldName='{name}']")

      if field.get('optype') != 'categorical':
        stats = bayesInput.find('TargetValueStats')
        targetValue = stats.find(f"TargetValueStat[@value='{target}']")
        distribution = targetValue.find('GaussianDistribution')

        if distribution is None:
          distributionName = targetValue.find('*').tag
          raise NotImplementedError(f'Distribution "{distributionName}" not implemented, or not supported '
                                    f'by scikit-learn')

        return [distribution]
      else:
        counts = [
          float(target_value_for_category(bayesInput, c).get('count'))
          for c in self.field_mapping[name][1].categories
        ]
        return [
          {
            'mean': count / np.sum(counts),
            'variance': 999999999
          }
          for count in counts
        ]

    return list(chain.from_iterable([
      target_value_for_field(name, field)
      for name, field in self.fields.items()
      if field is not self.target_field
    ]))

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return GaussianNB._more_tags(self)

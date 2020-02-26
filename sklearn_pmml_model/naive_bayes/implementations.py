from sklearn_pmml_model.base import PMMLBaseClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from itertools import chain


class PMMLGaussianNB(PMMLBaseClassifier, GaussianNB):
  """
  Abstract class for Naive Bayes models.

  """
  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)

    model = self.root.find('NaiveBayesModel')

    if model is None:
      raise Exception('PMML model does not contain NaiveBayesModel.')

    inputs = model.find('BayesInputs')

    target_values = {
      target: self._get_target_values(inputs, target)
      for target in self.classes_
    }

    self.class_prior_ = np.array([1 / len(self.classes_) for _ in self.classes_])
    self.theta_ = np.array([
      [float(value.get('mean', 0)) for value in target_values[target]]
      for target in self.classes_
    ])
    self.sigma_ = np.array([
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
          raise NotImplementedError(f'Distribution "{distributionName}" not implemented, or not supported by scikit-learn')

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

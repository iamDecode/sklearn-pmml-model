# License: BSD 2-Clause

import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn_pmml_model.base import PMMLBaseClassifier, PMMLBaseRegressor, IntegerEncodingMixin
from sklearn_pmml_model.tree import get_tree


class PMMLForestClassifier(IntegerEncodingMixin, PMMLBaseClassifier, RandomForestClassifier):
  """
  A random forest classifier.

  A random forest is a meta estimator that fits a number of decision tree
  classifiers on various sub-samples of the dataset and uses averaging to
  improve the predictive accuracy and control over-fitting.

  The PMML model consists out of a <Segmentation> element, that contains
  various <Segment> elements. Each segment contains it's own <TreeModel>.
  For Random Forests, only segments with a <True/> predicate are supported.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  n_jobs : int or None, optional (default=None)
      The number of jobs to run in parallel for the `predict` method.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/MultipleModels.html

  """

  def __init__(self, pmml, n_jobs=None):
    PMMLBaseClassifier.__init__(self, pmml)

    mining_model = self.root.find('MiningModel')
    if mining_model is None:
      raise Exception('PMML model does not contain MiningModel.')

    segmentation = mining_model.find('Segmentation')
    if segmentation is None:
      raise Exception('PMML model does not contain Segmentation.')

    if segmentation.get('multipleModelMethod') not in ['majorityVote', 'average']:
      raise Exception('PMML model ensemble should use majority vote or average.')

    # Parse segments
    segments = segmentation.findall('Segment')
    valid_segments = [segment for segment in segments if segment.find('True') is not None]

    if len(valid_segments) < len(segments):
      warnings.warn(
        'Warning: {} segment(s) ignored because of unsupported predicate.'
        .format(len(segments) - len(valid_segments))
      )

    n_estimators = len(valid_segments)
    RandomForestClassifier.__init__(self, n_estimators=n_estimators, n_jobs=n_jobs)
    self._validate_estimator()

    clf = self._make_estimator(append=False, random_state=123)
    clf.classes_ = self.classes_
    clf.n_features_ = self.n_features_
    clf.n_outputs_ = self.n_outputs_
    clf.n_classes_ = self.n_classes_
    self.template_estimator = clf

    self.estimators_ = [get_tree(self, s) for s in valid_segments]

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    for clf in self.estimators_:
      n_categories = np.asarray([
        len(self.field_mapping[field.get('name')][1].categories)
        if field.get('optype') == 'categorical' else -1
        for field in fields
        if field.tag == 'DataField'
      ], dtype=np.int32, order='C')
      clf.n_categories = n_categories
      clf.tree_.set_n_categories(n_categories)

    self.categorical = [x != -1 for x in self.estimators_[0].n_categories]

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)

  def _more_tags(self):
    return RandomForestClassifier._more_tags(self)


class PMMLForestRegressor(IntegerEncodingMixin, PMMLBaseRegressor, RandomForestRegressor):
  """
  A random forest regressor.

  A random forest is a meta estimator that fits a number of decision tree
  classifiers on various sub-samples of the dataset and uses averaging to
  improve the predictive accuracy and control over-fitting.

  The PMML model consists out of a <Segmentation> element, that contains
  various <Segment> elements. Each segment contains it's own <TreeModel>.
  For Random Forests, only segments with a <True/> predicate are supported.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  n_jobs : int or None, optional (default=None)
      The number of jobs to run in parallel for the `predict` method.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/MultipleModels.html

  """

  def __init__(self, pmml, n_jobs=None):
    PMMLBaseRegressor.__init__(self, pmml)

    mining_model = self.root.find('MiningModel')
    if mining_model is None:
      raise Exception('PMML model does not contain MiningModel.')

    segmentation = mining_model.find('Segmentation')
    if segmentation is None:
      raise Exception('PMML model does not contain Segmentation.')

    if segmentation.get('multipleModelMethod') not in ['majorityVote', 'average']:
      raise Exception('PMML model ensemble should use majority vote or average.')

    # Parse segments
    segments = segmentation.findall('Segment')
    valid_segments = [segment for segment in segments if segment.find('True') is not None]

    if len(valid_segments) < len(segments):
      warnings.warn(
        'Warning: {} segment(s) ignored because of unsupported predicate.'.format(
          len(segments) - len(valid_segments)
        )
      )

    n_estimators = len(valid_segments)
    self.n_outputs_ = 1
    RandomForestRegressor.__init__(self, n_estimators=n_estimators, n_jobs=n_jobs)
    self._validate_estimator()

    clf = self._make_estimator(append=False, random_state=123)
    clf.n_features_ = self.n_features_
    clf.n_outputs_ = self.n_outputs_
    self.template_estimator = clf

    self.estimators_ = [get_tree(self, s, rescale_factor=0.1) for s in valid_segments]

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    for clf in self.estimators_:
      n_categories = np.asarray([
        len(self.field_mapping[field.get('name')][1].categories)
        if field.get('optype') == 'categorical' else -1
        for field in fields
        if field.tag == 'DataField'
      ], dtype=np.int32, order='C')
      clf.n_categories = n_categories
      clf.tree_.set_n_categories(n_categories)

    self.categorical = [x != -1 for x in self.estimators_[0].n_categories]

  def fit(self, x, y):
    return PMMLBaseRegressor.fit(self, x, y)

  def _more_tags(self):
    return RandomForestRegressor._more_tags(self)

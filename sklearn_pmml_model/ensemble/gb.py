import numpy as np
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_pmml_model.base import PMMLBaseClassifier
from sklearn_pmml_model.tree import get_tree
from scipy.special import expit

class PMMLGradientBoostingClassifier(PMMLBaseClassifier, GradientBoostingClassifier):
  """
  Gradient Boosting for classification.

  GB builds an additive model in a  forward stage-wise fashion; it allows
  for the optimization of arbitrary differentiable loss functions. In each
  stage ``n_classes_`` regression trees are fit on the negative gradient of
  the binomial or multinomial deviance loss function. Binary classification
  is a special case where only a single regression tree is induced.

  The PMML model consists out of a <Segmentation> element, that contains
  various <Segment> elements. Each segment contains it's own <TreeModel>.
  For Gradient Boosting, only segments with a <True/> predicate are supported.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  Notes
  -----
  Specification: http://dmg.org/pmml/v4-3/MultipleModels.html

  """
  def __init__(self, pmml):
    PMMLBaseClassifier.__init__(self, pmml)

    mining_model = self.root.find('MiningModel')
    if mining_model is None:
      raise Exception('PMML model does not contain MiningModel.')

    segmentation = mining_model.find('Segmentation')
    if segmentation is None:
      raise Exception('PMML model does not contain Segmentation.')

    if segmentation.get('multipleModelMethod') not in ['modelChain']:
      raise Exception('PMML model ensemble should use modelChain.')

    # Parse segments
    segments = segmentation.findall('Segment')
    valid_segments = [None] * self.n_classes_

    indices = range(self.n_classes_)
    if self.n_classes_ == 2:
      indices = [0]

    if len(indices) == len(segments) - 1:
      for i in indices:
        valid_segments[i] = [segment for segment in segments[i].find('MiningModel').find('Segmentation').findall('Segment') if segment.find('True') is not None and segment.find('TreeModel') is not None]
    else:
      valid_segments = [[segment for segment in segments if segment.find('True') is not None and segment.find('TreeModel') is not None]]

    n_estimators = len(valid_segments[0])
    GradientBoostingClassifier.__init__(self, n_estimators=n_estimators)

    clf = DecisionTreeRegressor(random_state=123)
    clf.n_features_ = self.n_features_
    clf.n_outputs_ = self.n_outputs_
    self.template_estimator = clf

    self._check_params()

    try:
      self.init = None
      self._init_state()

      self.init_.class_prior_ = [expit(-float(segments[i].find('MiningModel').find('Targets').find('Target').get('rescaleConstant'))) for i in indices]

      if self.n_classes_ == 2:
        self.init_.class_prior_ = [self.init_.class_prior_[0], 1 - self.init_.class_prior_[0]]

      self.init_.classes_ = [i for i,_ in enumerate(self.classes_)]
      self.init_.n_classes_ = self.n_classes_
      self.init_.n_outputs_ = 1
      self.init_._strategy = self.init_.strategy
    except:
      self.init = 'zero'
      self._init_state()

    for x,y in np.ndindex(self.estimators_.shape):
      try:
        factor = float(segments[y].find('MiningModel').find('Targets').find('Target').get('rescaleFactor', 1))
        self.estimators_[x,y] = get_tree(self, valid_segments[y][x], rescale_factor=factor)
      except:
        self.estimators_[x,y] = get_tree(self, valid_segments[y][x])

    # Required after constructing trees, because categories may be inferred in
    # the parsing process
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]
    # for clf in self.estimators_[0]: # FIXME: should work for mutli class too
    #   n_categories = np.asarray([
    #     len(self.field_mapping[field.get('name')][1].categories)
    #     if field.get('optype') == 'categorical' else -1
    #     for field in fields
    #     if field.tag == 'DataField'
    #   ], dtype=np.int32, order='C')
    #   clf.n_categories = n_categories
    #   clf.tree_.set_n_categories(n_categories)

  def fit(self, x, y):
    return PMMLBaseClassifier.fit(self, x, y)
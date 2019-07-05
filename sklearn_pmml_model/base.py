from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from xml.etree import cElementTree as eTree
from cached_property import cached_property
from sklearn_pmml_model.datatypes import *
from collections import OrderedDict
import datetime
import numpy as np


class PMMLBaseEstimator(BaseEstimator):
  """
  Base class for estimators, saving basic information on DataFields.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  def __init__(self, pmml):
    it = eTree.iterparse(pmml)
    ns_offset = None
    for _, el in it:
      if ns_offset is None:
        ns_offset = el.tag.find('}') + 1
      el.tag = el.tag[ns_offset:]  # strip all namespaces
    self.root = it.root

    self.n_features_ = len([0 for e in self.fields.values() if e.tag == 'DataField']) - 1

  @cached_property
  def field_mapping(self):
    """
    Mapping from field name to column index and lambda function to process value.

    Returns
    -------
    { str: (int, callable) }
        Dictionary mapping column names to tuples with 1) index of the column
        and 2) type of the column.

    """
    target = self.target_field.get('name')
    fields = {name: field for name, field in self.fields.items() if name != target}
    field_labels = list(fields.keys())

    field_mapping = {
      name: (
        field_labels.index(name),
        get_type(e)
      )
      for name, e in fields.items()
      if e.tag == 'DataField'
    }

    field_mapping.update({
      name: (
        field_labels.index(e.find('FieldRef').get('field')),
        get_type(e, derives=fields[e.find('FieldRef').get('field')])
      )
      for name, e in fields.items()
      if e.tag == 'DerivedField'
    })

    field_mapping.update({
      self.target_field.get('name'): (
        None,
        get_type(self.target_field)
      )
    })

    return field_mapping

  @cached_property
  def fields(self):
    """
    Cached property containing an ordered mapping from field name to XML
    DataField or DerivedField element.

    Returns
    -------
    OrderedDict { str: eTree.Element }
        Where keys indicate field names, and values are XML elements.

    """
    data_dictionary = self.root.find('DataDictionary')
    transform_dict = self.root.find('TransformationDictionary')

    fields = OrderedDict({
      e.get('name'): e
      for e in findall(data_dictionary, 'DataField')
    })

    if transform_dict is not None:
      fields.update({
        e.get('name'): e
        for e in findall(transform_dict, 'DerivedField')
      })

    return fields

  @cached_property
  def target_field(self):
    """
    Cached property containing a reference to the XML DataField or DerivedField
    element corresponding to the classification target.

    Returns
    -------
    eTree.Element
        Representing the target field for classification, or None if no
        MiningSchema or MiningField specified.

    """
    mining_schema = next(self.root.iter('MiningSchema'), None)

    if mining_schema is not None:
      mining_field = next(
        (s for s in mining_schema if s.get('usageType') in ['target', 'predicted']),
        None
      )

      if mining_field is not None:
        return self.fields[mining_field.get('name')]

    return None

  def fit(self, x, y):
    """
    This method is not supported: PMML models are already fitted.

    """
    raise Exception('Not supported.')

  def _prepare_data(self, X):
    X = np.asarray(X)

    for column, (index, field_type) in self.field_mapping.items():
      if type(field_type) is Category and index is not None and type(X[0,index]) is str:
        categories = [str(v) for v in field_type.categories]
        categories += [c for c in np.unique(X[:,index]) if c not in categories]
        X[:,index] = [categories.index(x) for x in X[:,index]]

    return X

  def predict(self, X, *args, **kwargs):
    X = self._prepare_data(X)
    return super().predict(X, *args, **kwargs)

  def predict_proba(self, X, *args, **kwargs):
    X = self._prepare_data(X)
    return super().predict_proba(X, *args, **kwargs)


def get_type(data_field, derives=None):
  """
  Parse type defined in <DataField> object and returns it.

  Parameters
  ----------
  data_field : eTree.Element
      <DataField> or <DerivedField> XML element that describes a column.

  derives : eTree.Element
      <DataField> XML element that the derived field derives.

  Returns
  -------
  callable
      Type of the value, as a callable function.

  """
  data_type = data_field.get('dataType')

  type_mapping = {
    'string': str,
    'integer': int,
    'float': float,
    'double': float,
    'boolean': lambda x: x.lower() in ['1', 'true', 'yes'] if type(x) is str else bool(x),
    'date': datetime.date,
    'time': datetime.time,
    'dateTime': datetime.datetime,
    'dateDaysSince0': int,
    'dateDaysSince1960': int,
    'dateDaysSince1970': int,
    'dateDaysSince1980': int,
    'timeSeconds': int,
    'dateTimeSecondsSince0': int,
    'dateTimeSecondsSince1960': int,
    'dateTimeSecondsSince1970': int,
    'dateTimeSecondsSince1980': int,
  }

  if type_mapping.get(data_type) is None:
    raise Exception('Unsupported data type.')

  op_type = data_field.get('optype')

  if op_type not in ['categorical', 'ordinal', 'continuous']:
    raise Exception('Unsupported operation type.')

  if op_type == 'continuous':
    return type_mapping.get(data_type)
  else:
    categories = [
      e.get('value')
      for e in findall(data_field, 'Value') + findall(derives, 'Value')
    ]

    return Category(type_mapping[data_type], categories=categories, ordered=op_type == 'ordinal')


class PMMLBaseClassifier(PMMLBaseEstimator, ClassifierMixin):
  """
  Base class for classifiers, preparing classes, target fields

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  def __init__(self, pmml):
    super().__init__(pmml)

    target_type: Category = get_type(self.target_field)
    self.classes_ = np.array(target_type.categories)
    self.n_classes_ = len(self.classes_)
    self.n_outputs_ = 1


class PMMLBaseRegressor(PMMLBaseEstimator, RegressorMixin):
  """
  Base class for regressors.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """
  def __init__(self, pmml):
    super().__init__(pmml)


# Helper methods

def findall(element, path):
  if element is None:
    return []
  return element.findall(path)

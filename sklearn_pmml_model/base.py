# License: BSD 2-Clause

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xml.etree import cElementTree as eTree
from cached_property import cached_property
from sklearn_pmml_model.datatypes import Category
from sklearn.utils.multiclass import type_of_target
from collections import OrderedDict
import datetime
import re
import numpy as np
import pandas as pd


array_regex = re.compile(r"""('.*?'|".*?"|\S+)""")


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

    try:
      self.n_features_ = len([0 for e in self.fields.values() if e.tag == 'DataField']) - 1
    except AttributeError:
      self.n_features_in_ = len([0 for e in self.fields.values() if e.tag == 'DataField']) - 1

  @cached_property
  def field_mapping(self):
    """
    Map field name to a column index and lambda function that converts a value to the proper type.

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
      if e.tag == 'DerivedField' and e.find('FieldRef') is not None
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
    Return an ordered mapping from field name to XML DataField or DerivedField element.

    Returns
    -------
    OrderedDict { str: eTree.Element }
        Where keys indicate field names, and values are XML elements.

    """
    data_dictionary = self.root.find('DataDictionary')
    global_transforms = self.root.find('TransformationDictionary')
    local_transforms = self.root.findall('.//LocalTransformations')

    local_derived_fields = [findall(x, 'DerivedField') for x in local_transforms]
    local_derived_fields = [field for sublist in local_derived_fields for field in sublist]

    derived_fields = findall(global_transforms, 'DerivedField') + local_derived_fields

    fields = OrderedDict({
      e.get('name'): e
      for e in findall(data_dictionary, 'DataField')
    })

    if derived_fields:
      fields.update({
        e.get('name'): e
        for e in derived_fields
      })

    return fields

  @cached_property
  def target_field(self):
    """
    Return the XML DataField or DerivedField element corresponding to the classification target.

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
    """Not supported: PMML models are already fitted."""
    raise Exception('Not supported.')

  def _prepare_data(self, X):
    pmml_features = [f for f, e in self.fields.items() if e is not self.target_field and e.tag == 'DataField']

    if isinstance(X, pd.DataFrame):
      X.columns = X.columns.map(str)

      try:
        X = X[pmml_features]
      except KeyError:
        raise Exception('The features in the input data do not match features expected by the PMML model.')
    elif X.shape[1] != len(pmml_features):
      raise Exception('The number of features in provided data does not match expected number of features in the PMML. '
                      'Provide pandas.Dataframe, or provide data matching the DataFields in the PMML document.')

    return X

  def predict(self, X, *args, **kwargs):
    """
    Predict class or regression value for X.

    This call is preceded with a data preprocessing step, that enables data scaling
    and categorical feature encoding.

    For more information on parameters, check out the specific implementation in the
    scikit-learn subclass.

    """
    X = self._prepare_data(X)
    return super().predict(X, *args, **kwargs)

  def predict_proba(self, X, *args, **kwargs):
    """
    Predict class probabilities for X.

    This call is preceded with a data preprocessing step, that enables data scaling
    and categorical feature encoding.

    For more information on parameters, check out the specific implementation in the
    scikit-learn subclass.

    """
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


class PMMLBaseClassifier(PMMLBaseEstimator):
  """
  Base class for classifiers, preparing classes, target fields.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """

  def __init__(self, pmml):
    PMMLBaseEstimator.__init__(self, pmml)

    target_type: Category = get_type(self.target_field)
    try:
      self.classes_ = np.array(target_type.categories)
    except AttributeError:
      self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
      self._label_binarizer.classes_ = np.array(target_type.categories)
      self._label_binarizer.y_type_ = type_of_target(target_type.categories)
    self.n_classes_ = len(self.classes_)
    self.n_outputs_ = 1


class PMMLBaseRegressor(PMMLBaseEstimator):
  """
  Base class for regressors.

  Parameters
  ----------
  pmml : str, object
      Filename or file object containing PMML data.

  """

  def __init__(self, pmml):
    PMMLBaseEstimator.__init__(self, pmml)


# Helper methods

def findall(element, path):
  """Safe helper method to find XML elements with guaranteed return type."""
  if element is None:
    return []
  return element.findall(path)


def parse_array(array):
  """
  Convert <Array> or <SparseArray> element into list.

  Parameters
  ----------
  array : eTree.Element (Array or SparseArray)
      PMML <Array> or <SparseArray> element, or type-prefixed variant (e.g., <REAL-Array>).

  Returns
  -------
  output : list
    Python list containing the items described in the PMML array element.

  """
  tag = array.tag.lower()
  array_type = array.get('type', '').lower()

  def is_type(t):
    return tag.startswith(t) or array_type == t

  if tag.endswith('sparsearray'):
    return parse_sparse_array(array)

  if is_type('string'):
    # Deal with strings containing spaces wrapped in quotes (e.g., "like this")
    return [
      x.replace('"', '').replace('▲', '"')
      for x in array_regex.findall(array.text.replace('\\"', '▲'))
    ]

  if is_type('int'):
    return [int(x) for x in array.text.split(' ')]

  if is_type('num') or is_type('real') or is_type('prob') or is_type('percentage'):
    return [float(x) for x in array.text.split(' ')]

  raise Exception('Unknown array type encountered.')


def parse_sparse_array(array):
  """
  Convert <SparseArray> element into list.

  Parameters
  ----------
  array : eTree.Element (SparseArray)
      PMML <SparseArray> element, or type-prefixed variant (e.g., <REAL-SparseArray>).

  Returns
  -------
  output : list
    Python list containing the items described in the PMML sparse array element.

  """
  tag = array.tag.lower()
  array_type = array.get('type', '').lower()

  def is_type(t):
    return tag.startswith(t) or array_type == t

  values = [0] * int(array.get('n'))
  indices = [int(i) - 1 for i in array.find('Indices').text.split(' ')]
  entries = None

  element = array.find('Entries')

  if is_type('int'):
    if element is None:
      element = array.find('INT-Entries')

    entries = [int(x) for x in element.text.split(' ')]

  elif is_type('num') or is_type('real'):
    if element is None:
      element = array.find('NUM-Entries')
    if element is None:
      element = array.find('REAL-Entries')
    if element is None:
      raise Exception('Unknown array entries type encountered.')

    entries = [float(x) for x in element.text.split(' ')]

  else:
    raise Exception('Unknown array type encountered.')

  for index in indices:
    values[index] = entries[index]

  return values


class OneHotEncodingMixin:
  """Mixin class to automatically one-hot encode categorical variables."""

  def __init__(self):
    # Setup a column transformer to encode categorical variables
    target = self.target_field.get('name')
    fields = [field for name, field in self.fields.items() if name != target]

    def encoder_for(field):
      if field.get('optype') != 'categorical':
        return 'passthrough'

      encoder = OneHotEncoder()
      encoder.categories_ = np.array([self.field_mapping[field.get('name')][1].categories])
      encoder.drop_idx_ = np.array([None for x in encoder.categories_])
      encoder._legacy_mode = False
      return encoder

    transformer = ColumnTransformer(
      transformers=[
        (field.get('name'), encoder_for(field), [self.field_mapping[field.get('name')][0]])
        for field in fields
        if field.tag == 'DataField'
      ]
    )

    X = np.array([[0 for field in fields if field.tag == 'DataField']])
    transformer._validate_transformers()
    transformer._validate_column_callables(X)
    transformer._validate_remainder(X)
    transformer.transformers_ = transformer.transformers
    transformer.sparse_output_ = False
    transformer._feature_names_in = None

    self.transformer = transformer

  def _prepare_data(self, X):
    X = super()._prepare_data(X)
    return self.transformer.transform(X)


class IntegerEncodingMixin:
  """Mixin class to automatically integer encode categorical variables."""

  def _prepare_data(self, X):
    X = super()._prepare_data(X)
    X = np.asarray(X)

    for column, (index, field_type) in self.field_mapping.items():
      if type(field_type) is Category and index is not None and type(X[0, index]) is str:
        categories = [str(v) for v in field_type.categories]
        categories += [c for c in np.unique(X[:, index]) if c not in categories]
        X[:, index] = [categories.index(x) for x in X[:, index]]

    return X

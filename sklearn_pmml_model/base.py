from sklearn.base import BaseEstimator, ClassifierMixin
from xml.etree import cElementTree as eTree
from cached_property import cached_property
from sklearn_pmml_model.datatypes import *


class PMMLBaseEstimator(BaseEstimator, ClassifierMixin):
  def __init__(self, pmml):
    self.root = eTree.parse(pmml).getroot()
    self.namespace = self.root.tag[1:self.root.tag.index('}')]

  def find(self, element, path):
    return element.find(f"PMML:{path}", namespaces={"PMML": self.namespace})

  def findall(self, element, path):
    return element.findall(f"PMML:{path}", namespaces={"PMML": self.namespace})

  @cached_property
  def field_mapping(self):
    """
    Mapping from field name to column name and lambda function to process value.

    Returns
    -------
    dict {str: (str,lamba<pd.Series>)}
        Where keys indicate column names, and values are anonymous functions selecting the associated column from
        instance X, applying transformations defined in the pipeline and returning that value.

    """
    data_dictionary = self.find(self.root, 'DataDictionary')

    feature_mapping = {
      e.get('name'): (e.get('name'), lambda value, e=e: self.parse_type(value, e))
      for e in data_dictionary
      if e.tag == f'{{{self.namespace}}}DataField'
    }

    transformDict = self.find(self.root, 'TransformationDictionary')

    if transformDict is not None:
      feature_mapping.update({
        e.get('name'): (self.find(e, 'FieldRef').get('field'), lambda value, e=e: self.parse_type(value, e))
        for e in transformDict
        if e.tag == f'{{{self.namespace}}}DerivedField'
      })

    return feature_mapping

  def parse_type(self, value, data_field):
    """
    Parse type defined in <DataField> object, and convert value to that type.

    Parameters
    ----------
    value : Any
        Value that needs to be converted.

    data_field: xml.etree.ElementTree.Element
        <DataField> or <DerivedField> XML element that describes a column.

    Returns
    -------
    Any
        With same values as `value`, but converted to type defined in `dataField`.

    """
    # Check data type
    dataType = data_field.get('dataType')

    type_mapping = {
    # TODO: date, time, dateTime, dateDaysSince[0/1960/1970/1980], timeSeconds, dateTimeSecondsSince[0/1960/1970/1980]
      'string': str,
      'integer': int,
      'float': float,
      'double': float,
      'boolean': lambda x: Boolean(x.lower() in ['1', 'true', 'yes'] if type(x) is str else x)
    }

    if type_mapping.get(dataType) is None:
      raise Exception('Unsupported data type.')

    # Check operation type
    opType = data_field.get('optype')

    if opType not in ['categorical', 'ordinal', 'continuous']:
      raise Exception('Unsupported operation type.')

    value = type_mapping[dataType](value)

    # Check categories
    labels = [
      e.get('value')
      for e in data_field
      if e.tag == f'{{{self.namespace}}}Value'
    ]

    categories = [
      Category(label, labels, ordered=(opType == 'ordinal'))
      for label in labels
    ]

    intervals = [
      Interval(
        value=value,
        leftMargin=e.get('leftMargin'),
        rightMargin=e.get('rightMargin'),
        closure=e.get('closure')
      )
      for e in data_field
      if e.tag == f'{{{self.namespace}}}Interval'
    ]

    if len(categories) != 0:
      category = next((x for x in categories if x == value), None)

      if category is None:
        raise Exception('Value does not match any category.')
      else:
        return category

    if len(intervals) != 0:
      interval = next((x for x in intervals if value in x), None)

      if interval is None:
        raise Exception('Value does not match any interval.')
      else:
        interval.value = value
        return interval

    return value

  def fit(self, X, y):
    """
    This method is not supported: PMML models are already fitted.
    """
    raise Exception('Not supported.')

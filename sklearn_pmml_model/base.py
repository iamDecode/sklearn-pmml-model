from sklearn.base import BaseEstimator
from xml.etree import cElementTree as eTree
from cached_property import cached_property
from sklearn_pmml_model.datatypes import *
from collections import OrderedDict

class PMMLBaseEstimator(BaseEstimator):
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
    fields = self.fields
    if self.target_field is not None:
      del fields[self.target_field.get('name')]
    field_labels = list(self.fields.keys())

    field_mapping = {
      name: (
        field_labels.index(name),
        lambda value, e=e: self.parse_type(value, e)
      )
      for name, e in fields.items()
      if e.tag == f'{{{self.namespace}}}DataField'
    }

    field_mapping.update({
      name: (
        field_labels.index(self.find(e, 'FieldRef').get('field')),
        lambda value, e=e: self.parse_type(value, e)
      )
      for name, e in fields.items()
      if e.tag == f'{{{self.namespace}}}DerivedField'
    })

    return field_mapping

  @cached_property
  def fields(self):
    data_dictionary = self.find(self.root, 'DataDictionary')
    transform_dict = self.find(self.root, 'TransformationDictionary')

    fields = OrderedDict({
      e.get('name'): e
      for e in self.findall(data_dictionary, 'DataField')
    })

    if transform_dict is not None:
      fields.update({
        e.get('name'): e
        for e in self.findall(transform_dict, 'DerivedField')
      })

    return fields

  @cached_property
  def target_field(self):
    mining_schema = next(self.root.iter(f'{{{self.namespace}}}MiningSchema'), None)

    if mining_schema is not None:
      mining_field = next((s for s in mining_schema if s.get('usageType') in ['target', 'predicted']), None)

      if mining_field is not None:
        return self.fields[mining_field.get('name')]

    return None

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
      for e in self.findall(data_field, 'Value')
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
      for e in self.findall(data_field, 'Interval')
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

from sklearn.base import BaseEstimator
from xml.etree import cElementTree as eTree
from cached_property import cached_property
from sklearn_pmml_model.datatypes import *
from collections import OrderedDict


class PMMLBaseEstimator(BaseEstimator):
  def __init__(self, pmml, field_labels=None):
    self.root = eTree.parse(pmml).getroot()
    self.namespace = self.root.tag[1:self.root.tag.index('}')]
    self.field_labels = field_labels

  def find(self, element, path):
    if element is None: return None
    return element.find(f"PMML:{path}", namespaces={"PMML": self.namespace})

  def findall(self, element, path):
    if element is None: return []
    return element.findall(f"PMML:{path}", namespaces={"PMML": self.namespace})

  @cached_property
  def field_mapping(self):
    """
    Mapping from field name to column index and lambda function to process value.

    Returns
    -------
    dict { str: (int, lambda<pandas.Series>) }
        Where keys indicate column names, and values are anonymous functions selecting the associated column from
        instance X, applying transformations defined in the pipeline and returning that value.

    """
    target = self.target_field.get('name')
    fields = { name: field for name, field in self.fields.items() if name != target }
    field_labels = self.field_labels or list(fields.keys())

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
        lambda value, e=e, d=self.find(e, 'FieldRef').get('field'): self.parse_type(value, e, derives=fields[d])
      )
      for name, e in fields.items()
      if e.tag == f'{{{self.namespace}}}DerivedField'
    })

    field_mapping.update({
      self.target_field.get('name'): (
        None,
        lambda value, e=self.target_field: self.parse_type(value, e)
      )
    })

    return field_mapping

  @cached_property
  def fields(self):
    """
    Cached property containing an ordered mapping from field name to XML DataField or DerivedField element.

    Returns
    -------
    OrderedDict { str: eTree.Element }
        Where keys indicate field names, and values are XML elements.

    """
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
    """
    Cached property containing a reference to the XML DataField or DerivedField element corresponding to the
    classification target.

    Returns
    -------
    eTree.Element
        Representing the target field for classification, or None if no MiningSchema or MiningField specified.

    """
    mining_schema = next(self.root.iter(f'{{{self.namespace}}}MiningSchema'), None)

    if mining_schema is not None:
      mining_field = next((s for s in mining_schema if s.get('usageType') in ['target', 'predicted']), None)

      if mining_field is not None:
        return self.fields[mining_field.get('name')]

    return None

  def parse_type(self, value, data_field, derives=None, force_native=False):
    """
    Parse type defined in <DataField> object, and convert value to that type.

    Parameters
    ----------
    value : Any
        Value that needs to be converted.

    data_field : eTree.Element
        <DataField> or <DerivedField> XML element that describes a column.

    force_native : bool
        Boolean indicating whether native datatypes should be forced. Practically this means returning the base type of
        the field rather than sklearn_pmml_model datatype like Category or Interval.

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
      type_mapping[dataType](e.get('value'))
      for e in self.findall(data_field, 'Value') + self.findall(derives, 'Value')
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
      for e in self.findall(data_field, 'Interval') + self.findall(derives, 'Interval')
    ]

    if len(intervals) != 0:
      interval = next((x for x in intervals if value in x), None)

      if interval is None:
        raise Exception('Value does not match any interval.')
      else:
        interval.value = value
        return interval if not force_native else interval.value

    if opType != 'continuous':
      category = next((x for x in categories if x == value), None)

      if category is None:
        raise Exception('Value does not match any category.')
      else:
        return category if not force_native else category.value

    return value

  def fit(self, X, y):
    """
    This method is not supported: PMML models are already fitted.
    """
    raise Exception('Not supported.')

from sklearn.base import BaseEstimator
from xml.etree import cElementTree as etree
from cached_property import cached_property
import math
import operator as op


def find(element, path):
  return element.find(f"PMML:{path}", namespaces={"PMML": "http://www.dmg.org/PMML-4_3"})

def findall(element, path):
  return element.findall(f"PMML:{path}", namespaces={"PMML": "http://www.dmg.org/PMML-4_3"})


class Interval():
  def __init__(self, closure, leftMargin=None, rightMargin=None):
    if leftMargin is None and rightMargin is None:
      raise Exception("Interval not well defined.")

    self.closure = closure
    self.leftMargin = float(leftMargin or -math.inf)
    self.rightMargin = float(rightMargin or math.inf)

  def __eq__(self, other):
    if isinstance(other, float) or isinstance(other, int):
      closure_mapping = {
        'openClosed': [op.lt, op.le],
        'openOpen': [op.lt, op.lt],
        'closedOpen': [op.le, op.lt],
        'closedClosed': [op.le, op.le]
      }

      left, right = closure_mapping[self.closure]
      return left(self.leftMargin, other) and right(other, self.rightMargin)

    return self.leftMargin == other.leftMargin \
           and self.rightMargin == other.rightMargin \
           and self.closure == other.closure


class Category():
  def __init__(self, value, categories, ordered = False):
    if value not in categories:
      raise Exception("Invalid categorical value.")

    self.value = value
    self.categories = categories
    self.ordered = ordered

  def __eq__(self, other):
    return self.value == other

  def __lt__(self, other):
    if self.ordered:
      return self.categories.index(self) < self.categories.index(other)


class PMMLBaseEstimator(BaseEstimator):
  def __init__(self, pmml):
    self.root = etree.parse(pmml).getroot()

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
    dataDict = find(self.root, 'DataDictionary')

    feature_mapping = {
      e.get('name'): (e.get('name'), lambda value, e=e:
                                       self.parse_type(value, e))
      for e in dataDict
      if e.tag == '{http://www.dmg.org/PMML-4_3}DataField'
    }

    transformDict = find(self.root, 'TransformationDictionary')

    feature_mapping.update({
      e.get('name'): (find(e, 'FieldRef').get('field'), lambda value, e=e:
                                                          self.parse_type(value, e))
      for e in transformDict
      if e.tag == '{http://www.dmg.org/PMML-4_3}DerivedField'
    })

    return feature_mapping


  def parse_type(self, value, dataField):
    """
    Parse type defined in <DataField> object, and convert value to that type.

    Parameters
    ----------
    value : Any
        Value that needs to be converted.

    dataField: xml.etree.ElementTree.Element
        <DataField> or <DerivedField> XML element that describes a column.

    Returns
    -------
    Any
        With same values as `value`, but converted to type defined in `dataField`.

    """
    # Check data type
    dataType = dataField.get('dataType')

    type_mapping = { # TODO: date, time, dateTime, dateDaysSince[0/1960/1970/1980], timeSeconds, dateTimeSecondsSince[0/1960/1970/1980]
      'string': str,
      'integer': int,
      'float': float,
      'double': float,
      'boolean': bool
    }

    if type_mapping.get(dataType) is None:
      raise Exception("Unsupported data type.")

    # Check operation type
    opType = dataField.get('optype')

    if opType not in ['categorical', 'ordinal', 'continuous']:
      raise Exception("Unsupported operation type.")

    if opType == 'continuous':
      return type_mapping[dataType](value)

    else:
      labels = [
        e.get('value')
        for e in dataField
        if e.tag == '{http://www.dmg.org/PMML-4_3}Value'
      ]
      categories = [
        Category(label, labels, ordered = (opType == 'ordinal'))
        for label in labels
      ]
      # TODO: isCyclic

      categories += [
        Interval(
          leftMargin = e.get('leftMargin'),
          rightMargin = e.get('rightMargin'),
          closure = e.get('closure')
        )
        for e in dataField
        if e.tag == '{http://www.dmg.org/PMML-4_3}Interval'
      ]

      value = next((x for x in categories if x == value), None)

      if value is None:
        raise Exception(f"Invalid {opType} value.")

      return value


  def fit(self, X, y):
    """
    This method is not supported: PMML models are already fitted.
    """
    raise Exception("Not supported.")
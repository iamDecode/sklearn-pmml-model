import math
import operator as op


class Interval:
  def __init__(self, closure, leftMargin=None, rightMargin=None):
    assert leftMargin is not None or rightMargin is not None
    if leftMargin is not None and rightMargin is not None:
      assert leftMargin <= rightMargin
    assert closure in ['openClosed', 'openOpen', 'closedOpen', 'closedClosed']

    self.closure = closure
    self.leftMargin = float(leftMargin or -math.inf)
    self.rightMargin = float(rightMargin or math.inf)

  def __contains__(self, value):
    if isinstance(value, float) or isinstance(value, int):
      closure_mapping = {
        'openClosed': [op.lt, op.le],
        'openOpen': [op.lt, op.lt],
        'closedOpen': [op.le, op.lt],
        'closedClosed': [op.le, op.le]
      }

      left, right = closure_mapping[self.closure]
      return left(self.leftMargin, value) and right(value, self.rightMargin)


class Category:
  def __init__(self, base_type, categories, ordered = False):
    assert isinstance(categories, list)
    assert isinstance(ordered, bool)

    self.base_type = base_type

    self.categories = [base_type(cat) for cat in categories]
    self.ordered = ordered

  def __eq__(self, other):
    return type(other) == Category and \
           self.base_type == other.base_type and \
           self.categories == other.categories and \
           self.ordered == other.ordered

  def __contains__(self, item):
    return item in self.categories

  def __call__(self, value):
    value = self.base_type(value)

    if not value in self:
      raise Exception(f'Invalid categorical value: {value}')

    return value
import math
import operator as op


class Interval:
  def __init__(self, closure, left_margin=None, right_margin=None):
    assert left_margin is not None or right_margin is not None
    if left_margin is not None and right_margin is not None:
      assert left_margin <= right_margin
    assert closure in ['openClosed', 'openOpen', 'closedOpen', 'closedClosed']

    self.closure = closure
    self.leftMargin = float(left_margin or -math.inf)
    self.rightMargin = float(right_margin or math.inf)

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
  def __init__(self, base_type, categories, ordered=False):
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

    if value not in self:
      raise Exception(f'Invalid categorical value: {value}')

    return value

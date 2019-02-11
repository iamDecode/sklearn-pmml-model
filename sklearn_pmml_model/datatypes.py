import math
import operator as op

class PMMLDType:
  def __init__(self, value):
    self.value = value


class Interval(PMMLDType):
  def __init__(self, value, closure, leftMargin=None, rightMargin=None):
    assert leftMargin is not None or rightMargin is not None
    assert closure in ['openClosed', 'openOpen', 'closedOpen', 'closedClosed']

    super().__init__(value)

    self.closure = closure
    self.leftMargin = float(leftMargin or -math.inf)
    self.rightMargin = float(rightMargin or math.inf)

  def __eq__(self, other):
    if isinstance(other, Interval):
      return self.leftMargin == other.leftMargin \
             and self.rightMargin == other.rightMargin \
             and self.closure == other.closure \
             and self.value == other.value

    return other == self.value and other in self

  def __contains__(self, item):
    if isinstance(item, float) or isinstance(item, int):
      closure_mapping = {
        'openClosed': [op.lt, op.le],
        'openOpen': [op.lt, op.lt],
        'closedOpen': [op.le, op.lt],
        'closedClosed': [op.le, op.le]
      }

      left, right = closure_mapping[self.closure]
      return left(self.leftMargin, item) and right(item, self.rightMargin)


class Category(PMMLDType):
  def __init__(self, value, categories, ordered = False):
    assert isinstance(categories, list)
    assert isinstance(ordered, bool)

    if value not in categories:
      raise Exception('Invalid categorical value.')

    super().__init__(value)

    self.categories = categories
    self.ordered = ordered

  def __eq__(self, other):
    return self.value == other

  def __lt__(self, other):
    if self.ordered:
      return self.categories.index(self) < self.categories.index(other)
    raise Exception('Invalid operation for categorical value.')

  def __le__(self, other):
    if self.ordered:
      return self.categories.index(self) <= self.categories.index(other)
    raise Exception('Invalid operation for categorical value.')

  def __gt__(self, other):
    if self.ordered:
      return self.categories.index(self) > self.categories.index(other)
    raise Exception('Invalid operation for categorical value.')

  def __ge__(self, other):
    if self.ordered:
      return self.categories.index(self) >= self.categories.index(other)
    raise Exception('Invalid operation for categorical value.')


class Boolean(int, PMMLDType):
  def __new__(cls, value):
    self = int.__new__(cls, bool(value))
    PMMLDType.__init__(self, value)
    return self

  def __lt__(self, other):
    raise Exception('Invalid operation for Boolean value.')

  def __le__(self, other):
    raise Exception('Invalid operation for Boolean value.')

  def __gt__(self, other):
    raise Exception('Invalid operation for Boolean value.')

  def __ge__(self, other):
    raise Exception('Invalid operation for Boolean value.')

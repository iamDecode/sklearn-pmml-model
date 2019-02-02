from unittest import TestCase
from sklearn_pmml_model.datatypes import Category, Interval, Boolean


class TestInterval(TestCase):
  def test_exception(self):
    with self.assertRaises(Exception) as cm: Interval(1, closure='openOpen')
    assert type(cm.exception) == AssertionError

    with self.assertRaises(Exception) as cm: Interval(1, closure='non_existing_closure')
    assert type(cm.exception) == AssertionError

  def test_equation(self):
    a = Interval(8, 'closedClosed', 0, 10)
    b = Interval(9, 'closedClosed', 0, 10)
    c = Interval(9, 'closedClosed', 0, 10)

    assert a != b
    assert b == c

    assert 8 == a
    assert 9 == b
    assert 9 != a

class TestCategory(TestCase):
  def test_exception(self):
    with self.assertRaises(Exception) as cm: Category('1', [1, 2])
    assert str(cm.exception) == 'Invalid categorical value.'

  def test_invalid_operation(self):
    categories = ['value1', 'value2', 'value3']
    a = Category('value1', categories)
    b = Category('value2', categories)
    c = Category('value2', categories)

    assert a != b
    assert b == c

    with self.assertRaises(Exception) as cm: a > b
    lt = cm.exception

    with self.assertRaises(Exception) as cm: a >= b
    le = cm.exception

    with self.assertRaises(Exception) as cm: a < b
    gt = cm.exception

    with self.assertRaises(Exception) as cm: a <= b
    ge = cm.exception

    assert str(lt) == str(le) == str(gt) == str(ge) == 'Invalid operation for categorical value.'

  def test_ordinal_operation(self):
    categories = ['loud', 'louder', 'loudest']
    a = Category('loud', categories, ordered = True)
    b = Category('louder', categories, ordered = True)
    c = Category('louder', categories, ordered = True)

    assert a != b
    assert b == c

    assert b > a
    assert a < b
    assert b <= c
    assert b >= c

class TestBoolean(TestCase):
  def test_invalid_operation(self):
    a = Boolean(True)
    b = Boolean(False)
    c = Boolean(0)

    assert a != b
    assert b == c

    with self.assertRaises(Exception) as cm: a > b
    lt = cm.exception

    with self.assertRaises(Exception) as cm: a >= b
    le = cm.exception

    with self.assertRaises(Exception) as cm: a < b
    gt = cm.exception

    with self.assertRaises(Exception) as cm: a <= b
    ge = cm.exception

    assert str(lt) == str(le) == str(gt) == str(ge) == 'Invalid operation for Boolean value.'

from unittest import TestCase
from sklearn_pmml_model.datatypes import Category, Interval


class TestInterval(TestCase):
  def test_exception(self):
    with self.assertRaises(Exception) as cm:
      Interval(closure='openOpen')
    assert type(cm.exception) == AssertionError

    with self.assertRaises(Exception) as cm:
      Interval('openOpen', 3, 0)
    assert type(cm.exception) == AssertionError

    with self.assertRaises(Exception) as cm:
      Interval('non_existing_closure', 0)
    assert type(cm.exception) == AssertionError

  def test_contains(self):
    interval = Interval('closedClosed', 1, 10)

    assert 2 in interval
    assert 0 not in interval
    assert 10.1 not in interval


class TestCategory(TestCase):
  def test_exception(self):
    with self.assertRaises(Exception) as cm:
      Category(str, categories="bad cats")
    assert type(cm.exception) == AssertionError

    with self.assertRaises(Exception) as cm:
      Category(str, [1, 2], ordered=1)
    assert type(cm.exception) == AssertionError

  def test_contains(self):
    categories = ['loud', 'louder', 'loudest']
    cat_type = Category(str, categories, ordered=True)

    assert 'loud' in cat_type
    assert 'bad' not in cat_type

  def test_callable(self):
    categories = ['1', '2', '3']
    cat_type = Category(int, categories, ordered=True)

    with self.assertRaises(Exception) as cm:
      cat_type('4')

    assert str(cm.exception) == 'Invalid categorical value: 4'
    assert isinstance(cat_type('1'), int)
    assert cat_type('2') == 2

from unittest import TestCase
from sklearn_pmml_model.datatypes import Category, Interval

class TestDatatypes(TestCase):
  def test_interval_exception(self):
    with self.assertRaises(Exception): Interval(1, closure='openOpen')

  def test_category_exception(self):
    with self.assertRaises(Exception): Category('1', [1, 2])
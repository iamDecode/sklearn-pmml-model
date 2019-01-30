from unittest import TestCase
from sklearn_pmml_model.tree import PMMLTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import pandas as pd
import numpy as np
from os import path
from xml.etree import cElementTree as etree

# Parameters
pair = [0, 1]

# Load data
iris = load_iris()

# We only take the two corresponding features
X = pd.DataFrame(iris.data[:, pair])
X.columns = np.array(iris.feature_names)[pair]
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

df = pd.concat([Xte, yte], axis=1)


class TestTree(TestCase):
  def setUp(self):
    self.clf = PMMLTreeClassifier(pmml=StringIO("""
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="f1" optype="categorical" dataType="string">
          <Value value="value1"/>
          <Value value="value2"/>
          <Value value="value3"/>
        </DataField>
        <DataField name="f2" optype="ordinal" dataType="string">
          <Value value="loud"/>
          <Value value="louder"/>
          <Value value="loudest"/>
        </DataField>
        <DataField name="f3" optype="continuous" dataType="double">
          <Interval closure="closedClosed" leftMargin="4.3" rightMargin="7.9"/>
        </DataField>
        <DataField name="f4" optype="continuous" dataType="float"/>
        <DataField name="f5" optype="continuous" dataType="integer"/>
        <DataField name="f6" optype="continuous" dataType="boolean"/>
      </DataDictionary>
      <TreeModel/>
    </PMML>
    """))

  def test_evatuate_node(self):
    template = '<Node xmlns="http://www.dmg.org/PMML-4_3"><{}/></Node>'

    tests = {
      'True': True,
      'False': False,
      'SimplePredicate field="f1" operator="equal" value="value2"': True,
      'CompoundPredicate': Exception('Predicate not implemented'),
      'SimpleSetPredicate': Exception('Predicate not implemented'),
      'does_not_exist': False
    }

    instance = pd.Series(
      ['value2', 'louder', 4.5, 4.5, 4, True],
      index = ['f1','f2','f3','f4','f5','f6']
    )

    for element, expected in tests.items():
      node = etree.fromstring(template.format(element))

      if isinstance(expected, Exception):
        with self.assertRaises(Exception) as cm: self.clf.evaluate_node(node, instance)
        assert str(cm.exception) == str(expected)
      else:
        assert self.clf.evaluate_node(node, instance) == expected

  def test_evaluate_simple_predicate(self):
    template = '<SimplePredicate field="{}" operator="{}" value="{}"/>'

    tests = {
      ('f1', 'equal', 'value2'): True,
      ('f1', 'equal', 'value1'): False,
      ('f1', 'notEqual', 'value1'): True,
      ('f1', 'lessOrEqual', 'value2'): Exception("Invalid operation for categorical value."),
      ('f2', 'lessThan', 'loudest'): True,
      ('f2', 'greaterOrEqual', 'loud'): True,
      ('f2', 'greaterOrEqual', 'loudest'): False,
      ('f3', 'notEqual', 1.5): Exception("Value does not match any interval."),
      ('f3', 'equal', 4.4): False,
      ('f3', 'notEqual', 4.4): True,
      ('f3', 'equal', 4.5): True,
      ('f4', 'greaterOrEqual', 4.5): True,
      ('f4', 'greaterThan', 4.5): False,
      ('f4', 'greaterThan', 4.4): True,
      ('f5', 'lessThan', 5): True,
      ('f5', 'lessOrEqual', 3): False,
      ('f6', 'equal', 1): True,
      ('f6', 'equal', 0): False,
      ('f6', 'lessThan', 0): Exception("Invalid operation for Boolean value.")
    }

    instance = pd.Series(
      ['value2', 'louder', 4.5, 4.5, 4, True],
      index = ['f1','f2','f3','f4','f5','f6']
    )

    for attributes, expected in tests.items():
      element = etree.fromstring(template.format(*attributes))

      if isinstance(expected, Exception):
        with self.assertRaises(Exception) as cm: self.clf.evaluate_simple_predicate(element, instance)
        assert str(cm.exception) == str(expected)
      else:
        assert self.clf.evaluate_simple_predicate(element, instance) == expected

  def test_invalid_tree(self):
    with self.assertRaises(Exception) as cm:
      PMMLTreeClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>
      """))

    assert str(cm.exception) == 'PMML model does not contain TreeModel.'


class TestTreeIntegration(TestCase):
  def setUp(self):
    from sklearn_pmml_model.tree import PMMLTreeClassifier2
    self.clf2 = PMMLTreeClassifier2(pmml=path.join(path.dirname(__file__), '../models/sklearn2pmml.pmml'))
    self.clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/sklearn2pmml.pmml'))
    self.reference = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)

  def test_predict(self):
    assert np.array_equal(self.reference.predict(Xte), self.clf.predict(Xte))

  def test_predict_proba(self):
    assert np.array_equal(self.reference.predict_proba(Xte), self.clf.predict_proba(Xte))

  def test_score(self):
    assert self.reference.score(Xte, yte) == self.clf.score(Xte, yte)

  def test_predict_single(self):
    with self.assertRaises(Exception) as cm:
      a = self.clf.predict(Xte.iloc[0])
    exception = cm.exception

    with self.assertRaises(Exception) as cm:
      b = self.reference.predict(Xte.iloc[0])
    ref_exception = cm.exception

    assert str(exception) == str(ref_exception)

  def test_predict_proba_single(self):
    with self.assertRaises(Exception) as cm:
      a = self.clf.predict_proba(Xte.iloc[0])
    exception = cm.exception

    with self.assertRaises(Exception) as cm:
      b = self.reference.predict_proba(Xte.iloc[0])
    ref_exception = cm.exception

    assert str(exception) == str(ref_exception)


  def test_predict2(self):
    assert np.array_equal(self.reference.predict(Xte), self.clf2.predict(Xte))

  def test_predict_proba2(self):
    assert np.array_equal(self.reference.predict_proba(Xte), self.clf2.predict_proba(Xte))

  def test_score2(self):
     assert self.reference.score(Xte, yte) == self.clf2.score(Xte, yte)
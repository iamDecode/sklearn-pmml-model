from unittest import TestCase
from sklearn_pmml_model.tree import PMMLTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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


  def test_predict(self):
    clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/sklearn2pmml.pmml'))
    a = clf.predict(Xte)
    print(a)
    assert True


  # def test_predict_proba(self):
  #   a = self.clf.predict_proba(Xte)
  #   print(a)
  #   assert True
  #
  # def test_score(self):
  #   score = self.clf.score(Xte, yte)
  #   assert score  == 0.8
  #
  # def test_predict_dt(self):
  #   clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/lightpmmlpredictor.pmml'))
  #
  #   X = pd.DataFrame(data={
  #     'nom_nivell': ['ESO', 'CFGM Infor', 'ESO'],
  #     'hora_inici': ['09:15:00', '11:30:00', '09:15:00'],
  #     'assistenciaMateixaHora1WeekBefore': ['Present', 'Absent', 'NA'],
  #     'assistenciaMateixaHora2WeekBefore': ['NA', 'Absent', 'NA'],
  #     'assistenciaMateixaHora3WeekBefore': ['NA', 'Absent', 'NA'],
  #     'assistenciaaHoraAnterior': ['Present', 'Absent', 'NA']
  #   })
  #
  #   # ('Present', 0.9466557721489436)
  #   # ('Absent', 0.9154589371980676)
  #   # ('Present', 0.7301587301587301)
  #
  #   print(clf.predict(X))
  #   print(clf.predict_proba(X))

  def test_evaluate_simple_predicate(self):
    template = '<SimplePredicate field="{}" operator="{}" value="{}"/>'

    tests = {
      ('f1', 'equal', 'value2'): True,
      ('f1', 'notEqual', 'value1'): True,
      ('f1', 'lessOrEqual', 'value2'): Exception("Invalid operation for categorical value."),
      ('f2', 'lessThan', 'loudest'): True,
      ('f2', 'greaterOrEqual', 'loud'): True,
      ('f2', 'greaterOrEqual', 'loudest'): False,
      ('f3', 'notEqual', 1.5): Exception("Value does not match any interval."),
      ('f3', 'notEqual', 4.4): True,
      ('f3', 'equal', 4.5): True,
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
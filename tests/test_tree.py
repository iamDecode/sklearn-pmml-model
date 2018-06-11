from unittest import TestCase
from sklearn_pmml_model.tree import PMMLTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from os import path

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


feature_mapping_pmml = """
<PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
  <DataDictionary>
    <DataField name="Class" optype="categorical" dataType="string">
      <Value value="setosa"/>
      <Value value="versicolor"/>
      <Value value="virginica"/>
    </DataField>
    <DataField name="sepal length (cm)" optype="continuous" dataType="float"/>
    <DataField name="sepal width (cm)" optype="continuous" dataType="float"/>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="integer(sepal length (cm))" optype="continuous" dataType="integer">
      <FieldRef field="sepal length (cm)"/>
    </DerivedField>
    <DerivedField name="double(sepal width (cm))" optype="categorical" dataType="double">
      <FieldRef field="sepal width (cm)"/>
    </DerivedField>
  </TransformationDictionary>
  <TreeModel/>
</PMML>
"""

class TestTree(TestCase):

  def setUp(self):
    self.clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/sklearn2pmml.pmml'))


  def test_predict(self):
    a = self.clf.predict(Xte)
    print(a)
    assert True


  def test_predict_proba(self):
    a = self.clf.predict_proba(Xte)
    print(a)
    assert True

  def test_score(self):
    a = self.clf.score(Xte,yte)
    print(a)
    assert True

  def test_predict_dt(self):
    clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/lightpmmlpredictor.pmml'))

    X = pd.DataFrame(data={
      'nom_nivell': ['ESO', 'CFGM Infor', 'ESO'],
      'hora_inici': ['09:15:00', '11:30:00', '09:15:00'],
      'assistenciaMateixaHora1WeekBefore': ['Present', 'Absent', 'NA'],
      'assistenciaMateixaHora2WeekBefore': ['NA', 'Absent', 'NA'],
      'assistenciaMateixaHora3WeekBefore': ['NA', 'Absent', 'NA'],
      'assistenciaaHoraAnterior': ['Present', 'Absent', 'NA']
    })

    # ('Present', 0.9466557721489436)
    # ('Absent', 0.9154589371980676)
    # ('Present', 0.7301587301587301)

    print(clf.predict(X))
    print(clf.predict_proba(X))
from unittest import TestCase
from tree import PMMLTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from io import StringIO
from pandas.api.types import CategoricalDtype
from tree import find

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
    self.clf = PMMLTreeClassifier(pmml="../models/DecisionTreeIris.pmml")

  def test_predict(self):
    a = self.clf.predict(Xte)

    assert True

  def test_predict_proba(self):
    a = self.clf.predict_proba(Xte)

    assert True
from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.tree import PMMLTreeClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import pandas as pd
import numpy as np
from os import path, remove
from sklearn.datasets import load_digits


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestTree(TestCase):
  def test_invalid_tree(self):
    with self.assertRaises(Exception) as cm:
      PMMLTreeClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Class" optype="categorical" dataType="string">
            <Value value="setosa"/>
            <Value value="versicolor"/>
            <Value value="virginica"/>
          </DataField>
        </DataDictionary>
        <MiningSchema>
          <MiningField name="Class" usageType="target"/>
        </MiningSchema>
      </PMML>
      """))

    assert str(cm.exception) == 'PMML model does not contain TreeModel.'

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      pmml = path.join(BASE_DIR, '../models/decisionTree.pmml')
      clf = PMMLTreeClassifier(pmml=pmml)
      clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_incomplete_node(self):
    with self.assertRaises(Exception) as cm:
      PMMLTreeClassifier(pmml=StringIO("""
          <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
            <DataDictionary>
              <DataField name="Class" optype="categorical" dataType="string">
                <Value value="setosa"/>
                <Value value="versicolor"/>
                <Value value="virginica"/>
              </DataField>
            </DataDictionary>
            <MiningSchema>
              <MiningField name="Class" usageType="target"/>
            </MiningSchema>
            <TreeModel splitCharacteristic="binarySplit">
              <Node id="1">
                <True/>
                <Node id="2"/>
              </Node>
            </TreeModel>
          </PMML>
          """))

    assert str(cm.exception) == 'Node has insufficient information to ' \
                                'determine output: recordCount or score ' \
                                'attributes expected'

  def test_unsupported_predicate(self):
      with self.assertRaises(Exception) as cm:
        PMMLTreeClassifier(pmml=StringIO("""
        <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
          <DataDictionary>
            <DataField name="Class" optype="categorical" dataType="string">
              <Value value="setosa"/>
              <Value value="versicolor"/>
              <Value value="virginica"/>
            </DataField>
          </DataDictionary>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
          <TreeModel splitCharacteristic="binarySplit">
            <Node id="1">
              <True/>
              <Node id="2" score="setosa">
                <UnsupportedPredicate/>
              </Node>
              <Node id="3" score="versicolor">
                <UnsupportedPredicate/>
              </Node>
            </Node>
          </TreeModel>
        </PMML>
        """))

      assert str(cm.exception) == 'Unsupported tree format: unknown predicate' \
                                  ' structure in Node 2'

  def test_tree_threshold_value(self):
    clf = PMMLTreeClassifier(path.join(BASE_DIR, '../models/categorical.pmml'))
    assert clf.tree_.threshold == [[0, 4], 25.18735, -2, 125.5, -2, -2, 20.02033, -2, -2]


class TestIrisTreeIntegration(TestCase):
  def setUp(self):
    pair = [0, 1]
    data = load_iris()

    X = pd.DataFrame(data.data[:, pair])
    X.columns = np.array(data.feature_names)[pair]
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"
    X, Xte, y, yte = train_test_split(X, y, test_size=0.33, random_state=123)
    self.test = (Xte, yte)
    self.train = (X, y)

    pmml = path.join(BASE_DIR, '../models/decisionTree.pmml')
    self.clf = PMMLTreeClassifier(pmml=pmml)
    self.ref = DecisionTreeClassifier(random_state=1).fit(X, y)

  def test_predict(self):
    Xte, _ = self.test
    assert np.array_equal(self.ref.predict(Xte), self.clf.predict(Xte))

  def test_predict_proba(self):
    Xte, _ = self.test
    assert np.array_equal(
      self.ref.predict_proba(Xte),
      self.clf.predict_proba(Xte)
    )

  def test_score(self):
    Xte, yte = self.test
    assert self.ref.score(Xte, yte) == self.clf.score(Xte, yte)

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    pipeline.fit(self.train[0], self.train[1])
    sklearn2pmml(pipeline, "tree_sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLTreeClassifier(pmml='tree_sklearn2pmml.pmml')

      # Verify classification
      Xte, _ = self.test
      assert np.array_equal(
        self.ref.predict_proba(Xte),
        model.predict_proba(Xte)
      )

    finally:
      remove("tree_sklearn2pmml.pmml")



class TestDigitsTreeIntegration(TestCase):
  def setUp(self):
    data = load_digits()

    self.columns = [2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 17, 18, 19, 20, 21, 25, 26,
                    27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 41, 42, 43, 45, 46,
                    50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63]
    X = pd.DataFrame(data.data)
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"
    X, Xte, y, yte = train_test_split(X, y, test_size=0.33, random_state=123)
    self.test = (Xte, yte)

    self.clf = PMMLTreeClassifier(path.join(BASE_DIR, '../models/digits.pmml'))
    self.ref = DecisionTreeClassifier(random_state=1).fit(X, y)

  def test_predict(self):
    Xte, _ = self.test
    assert np.array_equal(
      self.ref.predict(Xte),
      self.clf.predict(Xte[self.columns]).astype(np.int64)
    )

  def test_predict_proba(self):
    Xte, _ = self.test
    assert np.array_equal(
      self.ref.predict_proba(Xte),
      self.clf.predict_proba(Xte[self.columns])
    )

  def test_score(self):
    Xte, yte = self.test
    assert self.ref.score(Xte, yte) == self.clf.score(Xte[self.columns], yte)


class TestCategoricalTreeIntegration(TestCase):
  def setUp(self):
    self.clf = PMMLTreeClassifier(path.join(BASE_DIR, '../models/cat.pmml'))

  def test_predict(self):
    Xte = np.array([[0], [1], [2]])
    assert np.array_equal(
      np.array(['class1', 'class2', 'class3']),
      self.clf.predict(Xte)
    )


class TestCategoricalPimaTreeIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats)
    df['age'] = df['age'].cat.codes
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/categorical.pmml')
    self.clf = PMMLTreeClassifier(pmml)

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([
      [0.1568627450980392, 0.84313725490196079],
      [0.7500000000000000, 0.25000000000000000],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.2000000000000000, 0.80000000000000004],
      [0.2000000000000000, 0.80000000000000004],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.2000000000000000, 0.80000000000000004],
      [0.9428571428571428, 0.05714285714285714],
      [0.2000000000000000, 0.80000000000000004],
      [0.2000000000000000, 0.80000000000000004],
      [0.9428571428571428, 0.05714285714285714],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.7368421052631579, 0.26315789473684209],
      [0.1568627450980392, 0.84313725490196079],
      [0.2000000000000000, 0.80000000000000004],
      [0.2000000000000000, 0.80000000000000004],
      [0.2000000000000000, 0.80000000000000004],
      [0.9428571428571428, 0.05714285714285714],
      [0.9428571428571428, 0.05714285714285714],
      [0.7368421052631579, 0.26315789473684209],
      [0.9428571428571428, 0.05714285714285714],
      [0.9428571428571428, 0.05714285714285714],
      [0.7368421052631579, 0.26315789473684209],
      [0.9428571428571428, 0.05714285714285714],
      [0.7368421052631579, 0.26315789473684209],
      [0.7500000000000000, 0.25000000000000000],
      [0.7368421052631579, 0.26315789473684209],
      [0.1568627450980392, 0.84313725490196079],
      [0.2000000000000000, 0.80000000000000004],
      [0.7368421052631579, 0.26315789473684209],
      [0.9428571428571428, 0.05714285714285714],
      [0.9428571428571428, 0.05714285714285714],
      [0.1568627450980392, 0.84313725490196079],
      [0.7368421052631579, 0.26315789473684209],
      [0.1568627450980392, 0.84313725490196079],
      [0.1568627450980392, 0.84313725490196079],
      [0.7368421052631579, 0.26315789473684209],
      [0.7368421052631579, 0.26315789473684209],
      [0.1568627450980392, 0.84313725490196079],
      [0.9428571428571428, 0.05714285714285714],
      [0.7368421052631579, 0.26315789473684209],
      [0.2000000000000000, 0.80000000000000004]
    ])
    assert np.array_equal(ref, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.7692307692307693
    assert ref == self.clf.score(Xte, yte)

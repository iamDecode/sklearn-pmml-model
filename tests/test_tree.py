from unittest import TestCase
from sklearn_pmml_model.tree import PMMLTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import pandas as pd
import numpy as np
from os import path
from sklearn.datasets import load_digits

class TestTree(TestCase):
  def test_invalid_tree(self):
    with self.assertRaises(Exception) as cm:
      PMMLTreeClassifier(pmml=StringIO("""<PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>"""))

    assert str(cm.exception) == 'PMML model does not contain TreeModel.'

  def test_non_binary_tree(self):
    with self.assertRaises(Exception) as cm:
      PMMLTreeClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <TreeModel splitCharacteristic="multiSplit" />
      </PMML>
      """))

    assert str(cm.exception) == 'Sklearn only supports binary classification models.'


class TestIrisTreeIntegration(TestCase):
  def setUp(self):
    pair = [0, 1]
    data = load_iris()

    X = pd.DataFrame(data.data[:, pair])
    X.columns = np.array(data.feature_names)[pair]
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)
    self.test = (Xte, yte)

    self.clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/sklearn2pmml.pmml'))
    self.reference = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)

  def test_predict(self):
    Xte, _ = self.test
    assert np.array_equal(self.reference.predict(Xte), self.clf.predict(Xte))

  def test_predict_proba(self):
    Xte, _ = self.test
    assert np.array_equal(self.reference.predict_proba(Xte), self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    assert self.reference.score(Xte, yte) == self.clf.score(Xte, yte)


class TestDigitsTreeIntegration(TestCase):
  def setUp(self):
    data = load_digits()

    X = pd.DataFrame(data.data)
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)
    self.test = (Xte, yte)

    self.clf = PMMLTreeClassifier(pmml=path.join(path.dirname(__file__), '../models/digits.pmml'), field_labels=np.array(X.columns).astype('U').tolist())
    self.reference = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)

  def test_predict(self):
    Xte, _ = self.test
    assert np.array_equal(self.reference.predict(Xte), self.clf.predict(Xte).astype(np.int64))

  def test_predict_proba(self):
    Xte, _ = self.test
    assert np.array_equal(self.reference.predict_proba(Xte), self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    assert self.reference.score(Xte, yte) == self.clf.score(Xte, yte)
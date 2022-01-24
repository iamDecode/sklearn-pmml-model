from unittest import TestCase
import sklearn_pmml_model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier, PMMLKNeighborsRegressor
import pandas as pd
import numpy as np
from os import path, remove
from io import StringIO
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestKNearestNeighbors(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLKNeighborsClassifier(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain NearestNeighborModel.'

  def test_no_distance_metric(self):
    with self.assertRaises(Exception) as cm:
      PMMLKNeighborsClassifier(pmml=StringIO("""
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
                <NearestNeighborModel numberOfNeighbors="5"/>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model does not contain ComparisonMeasure.'

  def test_unsupported_distance_metric(self):
    with self.assertRaises(Exception) as cm:
      PMMLKNeighborsClassifier(pmml=StringIO("""
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
                <NearestNeighborModel numberOfNeighbors="5">
                  <ComparisonMeasure>
                    <funkydistance/>
                  </ComparisonMeasure>
                </NearestNeighborModel>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model uses unsupported distance metric: "funkydistance".'


class TestKNeighborsClassifierIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats).codes + 1
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/knn-clf-pima.pmml')
    self.clf = PMMLKNeighborsClassifier(pmml)

  def test_predict(self):
    Xte, yte = self.test
    ref = np.array(['Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','No','No','No','Yes','No','No','No','No','No','No','No','Yes','No','No','No','No','No','No','Yes','No','No','No','No','Yes','Yes'])
    assert np.array_equal(ref, np.array(self.clf.predict(Xte)))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.807692307692307
    assert np.allclose(ref, self.clf.score(Xte, yte))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == KNeighborsClassifier()._more_tags()

  def test_sklearn2pmml(self):
    X, y = self.test
    ref = KNeighborsClassifier(n_neighbors=11)
    ref.fit(X, y)

    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", ref)
    ])
    pipeline.fit(self.test[0], self.test[1])
    sklearn2pmml(pipeline, "knn-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLKNeighborsClassifier(pmml='knn-sklearn2pmml.pmml')

      assert np.allclose(
        ref.predict_proba(X),
        model.predict_proba(X)
      )

    finally:
      remove("knn-sklearn2pmml.pmml")

  def test_sklearn2pmml_manhattan(self):
    X, y = self.test
    ref = KNeighborsClassifier(metric='manhattan', n_neighbors=8)
    ref.fit(X, y)

    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", ref)
    ])
    pipeline.fit(self.test[0], self.test[1])
    sklearn2pmml(pipeline, "knn-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLKNeighborsClassifier(pmml='knn-sklearn2pmml.pmml')

      assert np.allclose(
        ref.predict_proba(X),
        model.predict_proba(X)
      )

    finally:
      remove("knn-sklearn2pmml.pmml")


class TestKNeighborsRegressorIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats).codes + 1
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/knn-reg-pima.pmml')
    self.clf = PMMLKNeighborsRegressor(pmml)

  def test_predict(self):
    Xte, yte = self.test
    ref = np.array([
      0.7142857,
      0.1428571,
      0.7142857,
      1.0000000,
      0.8571429,
      0.8571429,
      0.7142857,
      0.8571429,
      0.2857143,
      0.8571429,
      0.8571429,
      0.7142857,
      0.8571429,
      1.0000000,
      0.8571429,
      0.7142857,
      0.1428571,
      1.0000000,
      0.5714286,
      0.7142857,
      0.8571429,
      0.5714286,
      0.4285714,
      0.5714286,
      1.0000000,
      0.5714286,
      0.8571429,
      0.1428571,
      0.1428571,
      0.2857143,
      0.7142857,
      0.1428571,
      0.2857143,
      0.0000000,
      0.1428571,
      0.2857143,
      0.1428571,
      0.0000000,
      0.8571429,
      0.4285714,
      0.2857143,
      0.1428571,
      0.1428571,
      0.1428571,
      0.1428571,
      0.7142857,
      0.0000000,
      0.1428571,
      0.2857143,
      0.1428571,
      0.7142857,
      0.7142857,
    ])
    assert np.allclose(ref, np.array(self.clf.predict(Xte)))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.383045525902668
    assert np.allclose(ref, self.clf.score(Xte, (yte == 'Yes').astype(int)))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == KNeighborsRegressor()._more_tags()

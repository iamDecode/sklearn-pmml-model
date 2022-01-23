from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.ensemble import PMMLForestClassifier, PMMLForestRegressor
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from io import StringIO
import numpy as np
from os import path, remove
import pandas as pd
from warnings import catch_warnings


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestForest(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLForestClassifier(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain MiningModel.'

  def test_invalid_segmentation(self):
    with self.assertRaises(Exception) as cm:
      PMMLForestClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Class" optype="categorical" dataType="string">
            <Value value="setosa"/>
            <Value value="versicolor"/>
            <Value value="virginica"/>
          </DataField>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
        </MiningModel>
      </PMML>
      """))

    assert str(cm.exception) == 'PMML model does not contain Segmentation.'

  def test_non_voting_ensemble(self):
    with self.assertRaises(Exception) as cm:
      PMMLForestClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Class" optype="categorical" dataType="string">
            <Value value="setosa"/>
            <Value value="versicolor"/>
            <Value value="virginica"/>
          </DataField>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
          <Segmentation multipleModelMethod="mean" />
        </MiningModel>
      </PMML>
      """))

    assert str(cm.exception) == 'PMML model ensemble should use majority vote or average.'

  def test_non_true_segment(self):
    with self.assertRaises(Exception), catch_warnings(record=True) as w:
      PMMLForestClassifier(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Class" optype="categorical" dataType="string">
            <Value value="setosa"/>
            <Value value="versicolor"/>
            <Value value="virginica"/>
          </DataField>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
          <Segmentation multipleModelMethod="majorityVote">
            <Segment>
              <False/>
            </Segment>
            <Segment>
              <True/>
            </Segment>
          </Segmentation>
        </MiningModel>
      </PMML>
      """))
    assert len([w for w in w if not isinstance(w.message, FutureWarning)]) == 1

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      pmml = path.join(BASE_DIR, '../models/rf-cat-pima.pmml')
      clf = PMMLForestClassifier(pmml)
      clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
      clf = PMMLForestClassifier(path.join(BASE_DIR, '../models/rf-cat-pima.pmml'))
      assert clf._more_tags() == RandomForestClassifier()._more_tags()


class TestForestRegression(TestCase):
    def test_invalid_model(self):
        with self.assertRaises(Exception) as cm:
            PMMLForestRegressor(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Output" optype="continuous" dataType="double"/>
        </DataDictionary>
        <MiningSchema>
          <MiningField name="Output" usageType="target"/>
        </MiningSchema>
      </PMML>
      """))

        assert str(cm.exception) == 'PMML model does not contain MiningModel.'

    def test_invalid_segmentation(self):
        with self.assertRaises(Exception) as cm:
            PMMLForestRegressor(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Output" optype="continuous" dataType="double"/>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Output" usageType="target"/>
          </MiningSchema>
        </MiningModel>
      </PMML>
      """))

        assert str(cm.exception) == 'PMML model does not contain Segmentation.'

    def test_non_voting_ensemble(self):
        with self.assertRaises(Exception) as cm:
            PMMLForestRegressor(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Output" optype="continuous" dataType="double"/>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Output" usageType="target"/>
          </MiningSchema>
          <Segmentation multipleModelMethod="mean" />
        </MiningModel>
      </PMML>
      """))

        assert str(cm.exception) == 'PMML model ensemble should use majority vote or average.'

    def test_non_true_segment(self):
        with self.assertRaises(Exception), catch_warnings(record=True) as w:
            PMMLForestRegressor(pmml=StringIO("""
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Output" optype="continuous" dataType="double"/>
        </DataDictionary>
        <MiningModel>
          <MiningSchema>
            <MiningField name="Output" usageType="target"/>
          </MiningSchema>
          <Segmentation multipleModelMethod="majorityVote">
            <Segment>
              <False/>
            </Segment>
            <Segment>
              <True/>
            </Segment>
          </Segmentation>
        </MiningModel>
      </PMML>
      """))
        assert len([w for w in w if not isinstance(w.message, FutureWarning)]) == 1

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            pmml = path.join(BASE_DIR, '../models/rf-cat-pima-regression.pmml')
            clf = PMMLForestRegressor(pmml)
            clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        clf = PMMLForestRegressor(path.join(BASE_DIR, '../models/rf-cat-pima-regression.pmml'))
        assert clf._more_tags() == RandomForestRegressor()._more_tags()


class TestIrisForestIntegration(TestCase):
  def setUp(self):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data)
    X.columns = np.array(['sepal_length', 'sepal_width', 'petal_length',
                          'petal_width'])
    y = pd.Series(np.array(np.array(['setosa', 'versicolor', 'virginica']))[iris.target])
    y.name = "Class"
    self.test = X, y

    self.clf = PMMLForestClassifier(path.join(BASE_DIR, '../models/rf-iris.pmml'))
    self.ref = RandomForestClassifier(random_state=1).fit(X, y)

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([
        [1., 0., 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.985, 0.015, 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.99, 0.01, 0.],
        [1., 0., 0.],
        [0.995, 0.005, 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.995, 0.005, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.98, 0.02, 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0.98, 0.015, 0.005],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0.995, 0.005],
        [0., 1., 0.],
        [0., 0.595, 0.405],
        [0., 0.995, 0.005],
        [0., 0.99, 0.01],
        [0., 0.995, 0.005],
        [0., 0.97, 0.03],
        [0.005, 0.805, 0.19],
        [0., 1., 0.],
        [0.005, 0.985, 0.01],
        [0., 0.95, 0.05],
        [0., 0.995, 0.005],
        [0., 0.985, 0.015],
        [0., 0.995, 0.005],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0.995, 0.005],
        [0., 1., 0.],
        [0., 0.99, 0.01],
        [0., 0.995, 0.005],
        [0., 0.645, 0.355],
        [0., 1., 0.],
        [0., 0.76, 0.24],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0.85, 0.15],
        [0., 0.025, 0.975],
        [0., 0.995, 0.005],
        [0., 0.995, 0.005],
        [0., 0.995, 0.005],
        [0., 0.995, 0.005],
        [0., 1., 0.],
        [0., 0.085, 0.915],
        [0.025, 0.96, 0.015],
        [0.01, 0.945, 0.045],
        [0., 1., 0.],
        [0., 0.995, 0.005],
        [0., 1., 0.],
        [0., 0.995, 0.005],
        [0., 0.98, 0.02],
        [0., 0.995, 0.005],
        [0., 0.99, 0.01],
        [0., 0.955, 0.045],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0.99, 0.01],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0.01, 0.99],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.245, 0.755],
        [0., 0., 1.],
        [0., 0.065, 0.935],
        [0., 0., 1.],
        [0., 0.01, 0.99],
        [0., 0., 1.],
        [0., 0.005, 0.995],
        [0., 0.055, 0.945],
        [0., 0.005, 0.995],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.015, 0.985],
        [0., 0.19, 0.81],
        [0., 0., 1.],
        [0., 0.11, 0.89],
        [0., 0., 1.],
        [0., 0.07, 0.93],
        [0., 0., 1.],
        [0., 0.005, 0.995],
        [0., 0.12, 0.88],
        [0., 0.035, 0.965],
        [0., 0., 1.],
        [0., 0.05, 0.95],
        [0., 0., 1.],
        [0., 0.005, 0.995],
        [0., 0., 1.],
        [0., 0.115, 0.885],
        [0., 0.09, 0.91],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.18, 0.82],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.005, 0.995],
        [0., 0.01, 0.99],
        [0., 0.005, 0.995],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.09, 0.91],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0.025, 0.975]
    ])

    assert np.array_equal(ref, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.9866666666666667
    assert ref == self.clf.score(Xte, yte)

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    pipeline.fit(self.test[0], self.test[1])

    sklearn2pmml(pipeline, "rf-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLForestClassifier(pmml='rf-sklearn2pmml.pmml')

      # Verify classification
      Xte, _ = self.test
      assert np.array_equal(
        self.ref.predict_proba(Xte),
        model.predict_proba(Xte)
      )

    finally:
      remove("rf-sklearn2pmml.pmml")


class TestCategoricalPimaForestIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats)
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/rf-cat-pima.pmml')
    self.clf = PMMLForestClassifier(pmml)

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([
      [0.2200000000000000, 0.7800000000000000],
      [0.5800000000000000, 0.4200000000000000],
      [0.2200000000000000, 0.7800000000000000],
      [0.1800000000000000, 0.8200000000000000],
      [0.0800000000000000, 0.9200000000000000],
      [0.3000000000000000, 0.7000000000000000],
      [0.2600000000000000, 0.7400000000000000],
      [0.3600000000000000, 0.6400000000000000],
      [0.4000000000000000, 0.6000000000000000],
      [0.1600000000000000, 0.8400000000000000],
      [0.2200000000000000, 0.7800000000000000],
      [0.3000000000000000, 0.7000000000000000],
      [0.1400000000000000, 0.8600000000000000],
      [0.6400000000000000, 0.3600000000000000],
      [0.1800000000000000, 0.8200000000000000],
      [0.1800000000000000, 0.8200000000000000],
      [0.7600000000000000, 0.2400000000000000],
      [0.2400000000000000, 0.7600000000000000],
      [0.3400000000000000, 0.6600000000000000],
      [0.2800000000000000, 0.7200000000000000],
      [0.0800000000000000, 0.9200000000000000],
      [0.2000000000000000, 0.8000000000000000],
      [0.6800000000000000, 0.3200000000000000],
      [0.1200000000000000, 0.8800000000000000],
      [0.2200000000000000, 0.7800000000000000],
      [0.3600000000000000, 0.6400000000000000],
      [0.2000000000000000, 0.8000000000000000],
      [0.8600000000000000, 0.1400000000000000],
      [0.9399999999999999, 0.0600000000000000],
      [0.7200000000000000, 0.2800000000000000],
      [0.5600000000000001, 0.4400000000000000],
      [0.9800000000000000, 0.0200000000000000],
      [0.4400000000000000, 0.5600000000000001],
      [0.8800000000000000, 0.1200000000000000],
      [0.6600000000000000, 0.3400000000000000],
      [0.5000000000000000, 0.5000000000000000],
      [0.7400000000000000, 0.2600000000000000],
      [0.2600000000000000, 0.7400000000000000],
      [0.1600000000000000, 0.8400000000000000],
      [0.6800000000000000, 0.3200000000000000],
      [0.7600000000000000, 0.2400000000000000],
      [0.7400000000000000, 0.2600000000000000],
      [0.5600000000000001, 0.4400000000000000],
      [0.5400000000000000, 0.4600000000000000],
      [0.5200000000000000, 0.4800000000000000],
      [0.1400000000000000, 0.8600000000000000],
      [0.7600000000000000, 0.2400000000000000],
      [0.8200000000000000, 0.1800000000000000],
      [0.4400000000000000, 0.5600000000000001],
      [0.9200000000000000, 0.0800000000000000],
      [0.5600000000000001, 0.4400000000000000],
      [0.2800000000000000, 0.7200000000000000]
    ])
    assert np.allclose(ref, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.7884615384615384
    assert ref == self.clf.score(Xte, yte)


class TestCategoricalPimaForestRegressionIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        cats = np.unique(df['age'])
        df['age'] = pd.Categorical(df['age'], categories=cats)
        Xte = df.iloc[:, 1:]
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)

        pmml = path.join(BASE_DIR, '../models/rf-cat-pima-regression.pmml')
        self.clf = PMMLForestRegressor(pmml)

    def test_predict_proba(self):
        Xte, _ = self.test
        ref = np.array([0.8928571428571429, 0.6785714285714286, 0.9642857142857143, 0.8571428571428571, 1.0000000000000000, 1.0000000000000000, 0.7619047619047619, 0.8571428571428571, 0.5833333333333333, 1.0000000000000000, 1.0000000000000000, 0.8928571428571429, 0.9000000000000000, 0.9714285714285714, 1.0000000000000000, 0.9714285714285714, 0.3452380952380952, 0.6785714285714286, 0.9047619047619048, 0.8571428571428571, 0.9642857142857143, 0.7142857142857143, 0.8214285714285714, 0.8095238095238095, 0.6500000000000000, 0.8571428571428571, 0.2500000000000000, 0.1428571428571428, 0.0000000000000000, 0.4285714285714285, 0.1785714285714286, 0.0000000000000000, 0.2500000000000000, 0.0000000000000000, 0.0000000000000000, 0.1547619047619047, 0.0952380952380952, 0.0000000000000000, 0.4952380952380952, 0.3214285714285715, 0.1071428571428571, 0.0000000000000000, 0.2500000000000000, 0.0000000000000000, 0.1785714285714286, 0.1142857142857143, 0.0000000000000000, 0.0000000000000000, 0.3214285714285715, 0.1428571428571428, 0.3809523809523809, 0.0476190476190476])
        assert np.allclose(ref, self.clf.predict(Xte))

    def test_score(self):
        Xte, yte = self.test
        ref = 0.8155071515785801
        assert ref == self.clf.score(Xte, yte == "Yes")
from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.naive_bayes import PMMLGaussianNB
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from os import path
from io import StringIO


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestNaiveBayes(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLGaussianNB(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
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

    assert str(cm.exception) == 'PMML model does not contain NaiveBayesModel.'

  def test_unsupported_distribution(self):
    with self.assertRaises(Exception) as cm:
      PMMLGaussianNB(pmml=StringIO("""
                <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                  <DataDictionary>
                    <DataField name="Class" optype="categorical" dataType="string">
                      <Value value="setosa"/>
                      <Value value="versicolor"/>
                      <Value value="virginica"/>
                    </DataField>
                    <DataField name="a" optype="continuous" dataType="double"/>
                  </DataDictionary>
                  <NaiveBayesModel>
                    <MiningSchema>
                      <MiningField name="Class" usageType="target"/>
                    </MiningSchema>
                    <BayesInputs>
                      <BayesInput fieldName="a">
                        <TargetValueStats>
                          <TargetValueStat value="setosa">
                            <PoissonDistribution mean="2.80188679245283"/>
                          </TargetValueStat>
                        </TargetValueStats>
                      </BayesInput>
                    </BayesInputs>
                  </NaiveBayesModel>
                </PMML>
                """))

    assert str(cm.exception) == 'Distribution "PoissonDistribution" not implemented, or not supported by scikit-learn'


class TestGaussianNBIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xte = pd.get_dummies(Xte, prefix_sep='')
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/nb-cat-pima.pmml')
    self.clf = PMMLGaussianNB(pmml)

    ref = GaussianNB()
    ref.fit(Xte, yte)
    print(ref)

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([0.089665518, 0.229009345, 0.007881006, 0.025306284, 0.013287187, 0.085741556, 0.338780868, 0.063463670, 0.769219497, 0.100369704, 0.002308186, 0.050380836, 0.054716302, 0.114718523, 0.156496072, 0.076301905, 0.806474996, 0.001227284, 0.121921194, 0.146751623, 0.074212037, 0.084148702, 0.479980587, 0.234470483, 0.354876655, 0.480582547, 0.113901660, 0.969566830, 0.989918477, 0.760519487, 0.599039599, 0.997856475, 0.776102648, 0.863233887, 0.910001902, 0.846005607, 0.734269347, 0.841546008, 0.120615475, 0.457027577, 0.124201960, 0.882691224, 0.930458760, 0.585210046, 0.484105369, 0.697949034, 0.778448666, 0.820806942, 0.074380668, 0.978478762, 0.589284915, 0.586728917])
    assert np.allclose(ref, self.clf.predict_proba(Xte)[:, 0])

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array(['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No'])
    assert all(ref == self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.8461538462
    assert np.allclose(ref, self.clf.score(Xte, yte))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

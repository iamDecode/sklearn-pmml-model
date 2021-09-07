from unittest import TestCase

from sklearn.datasets import load_iris

import sklearn_pmml_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, Lasso, ElasticNet
from sklearn_pmml_model.linear_model import PMMLLinearRegression, PMMLLogisticRegression, PMMLRidge, PMMLRidgeClassifier, PMMLLasso, PMMLElasticNet
import pandas as pd
import numpy as np
from os import path, remove
from io import StringIO
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestLinearRegression(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLLinearRegression(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain RegressionModel.'

  def test_nonlinear_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLLinearRegression(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="a" optype="continuous" dataType="double"/>
                </DataDictionary>
                <RegressionModel>
                  <MiningSchema>
                    <MiningField name="Class" usageType="target"/>
                  </MiningSchema>
                  <RegressionTable>
                    <NumericPredictor name="a" exponent="1" coefficient="1"/>
                    <NumericPredictor name="a" exponent="1" coefficient="1"/>
                  </RegressionTable>
                </RegressionModel>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model is not linear.'


class TestLogisticRegression(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLLogisticRegression(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain RegressionModel or Segmentation.'

  def test_nonlinear_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLLogisticRegression(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="a" optype="continuous" dataType="double"/>
                </DataDictionary>
                <RegressionModel>
                  <MiningSchema>
                    <MiningField name="Class" usageType="target"/>
                  </MiningSchema>
                  <RegressionTable>
                    <NumericPredictor name="a" exponent="1" coefficient="1"/>
                    <NumericPredictor name="a" exponent="1" coefficient="1"/>
                  </RegressionTable>
                </RegressionModel>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model is not linear.'

  def test_non_modelchain_segmentation(self):
    with self.assertRaises(Exception) as cm:
      PMMLLogisticRegression(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="a" optype="continuous" dataType="double"/>
                </DataDictionary>
                <MiningSchema>
                  <MiningField name="Class" usageType="target"/>
                </MiningSchema>
                <MiningModel>
                  <Segmentation multipleModelMethod="notModelChain" />
                </MiningModel>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model for multi-class logistic regression should use modelChain method.'


class TestLinearRegressionIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-lm.pmml')
    self.clf = PMMLLinearRegression(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array([0.959047661, 0.306635481, 0.651029985, 0.954874880, 0.889268596, 0.874413539, 0.552911965, 0.793971929, 0.567604727, 0.694441984, 0.977588079, 1.020076443, 0.938209348, 0.741296266, 0.785681506, 0.783314305, 0.147203243, 0.953499858, 0.861694209, 0.818535888, 1.054586791, 0.508564304, 0.490740907, 0.692194962, 0.546339084, 0.584074930, 0.817451147, 0.007120341, -0.023103301, 0.354232979, 0.452602313, -0.232817829, 0.289612034, 0.241502904, 0.098388728, 0.413283786, 0.349024715, 0.315999598, 0.656973238, 0.525739661, 0.243258999, 0.128203855, 0.151826018, 0.357043960, 0.647876971, 0.405659892, 0.264334997, 0.280004394, 0.948749766, -0.028252457, 0.415301011, 0.509803923])
    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.409378064635437
    assert np.allclose(ref, self.clf.score(Xte, yte == 'Yes'))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == LinearRegression()._more_tags()


class TestLogisticRegressionIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xte = pd.get_dummies(Xte, prefix_sep='')
    del Xte['age(20,30]']
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-lmc.pmml')
    self.clf = PMMLLogisticRegression(pmml)

    self.ref = LogisticRegression()
    self.ref.fit(Xte, yte)

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([
      [0.3836644757747519, 0.6163355242252481],
      [0.5572948024306759, 0.4427051975693241],
      [0.2208655069363070, 0.7791344930636930],
      [0.1226755439085095, 0.8773244560914905],
      [0.1116580306897623, 0.8883419693102377],
      [0.2319071635514390, 0.7680928364485610],
      [0.3884229315951135, 0.6115770684048865],
      [0.2465287129542991, 0.7534712870457009],
      [0.6593253655911793, 0.3406746344088207],
      [0.2374749130836621, 0.7625250869163379],
      [0.1540677632287771, 0.8459322367712229],
      [0.3435398902933879, 0.6564601097066121],
      [0.1625519980431368, 0.8374480019568632],
      [0.1469729880397515, 0.8530270119602485],
      [0.2418801127109025, 0.7581198872890975],
      [0.3625854866670420, 0.6374145133329580],
      [0.7838396580528175, 0.2161603419471824],
      [0.1327352123858896, 0.8672647876141104],
      [0.4539315153105434, 0.5460684846894566],
      [0.2653170373440615, 0.7346829626559385],
      [0.2214510855011292, 0.7785489144988708],
      [0.3815617982317231, 0.6184382017682769],
      [0.5460774966173132, 0.4539225033826867],
      [0.4194973709123712, 0.5805026290876288],
      [0.5259283752311108, 0.4740716247688892],
      [0.3775730286932922, 0.6224269713067078],
      [0.3428809879226986, 0.6571190120773014],
      [0.7310789777058304, 0.2689210222941696],
      [0.7808228627650035, 0.2191771372349965],
      [0.6285876142112172, 0.3714123857887828],
      [0.4444194848409649, 0.5555805151590351],
      [0.9253101654677492, 0.0746898345322508],
      [0.7108575012260019, 0.2891424987739981],
      [0.6923236892085397, 0.3076763107914603],
      [0.8658398719063449, 0.1341601280936550],
      [0.6859457061731435, 0.3140542938268565],
      [0.7190807857278905, 0.2809192142721096],
      [0.7983745824241288, 0.2016254175758712],
      [0.4768552271854714, 0.5231447728145286],
      [0.5321014224575110, 0.4678985775424890],
      [0.4293961594534983, 0.5706038405465017],
      [0.6791961042739789, 0.3208038957260211],
      [0.8898092885722800, 0.1101907114277199],
      [0.6579005184496933, 0.3420994815503067],
      [0.7652153232481362, 0.2347846767518638],
      [0.5160770109846871, 0.4839229890153129],
      [0.8067983092623874, 0.1932016907376126],
      [0.7877539634640341, 0.2122460365359659],
      [0.6347873007218796, 0.3652126992781204],
      [0.8190797854627907, 0.1809202145372093],
      [0.5519351414476166, 0.4480648585523834],
      [0.4482439620440842, 0.5517560379559158],
    ])
    assert np.allclose(ref, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.8076923076923077
    assert np.allclose(ref, self.clf.score(Xte, yte))

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    pipeline.fit(self.test[0], self.test[1])
    sklearn2pmml(pipeline, "lmc-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLLogisticRegression(pmml='lmc-sklearn2pmml.pmml')

      # Verify classification
      Xenc, _ = self.test
      assert np.allclose(
        self.ref.predict_proba(Xenc),
        model.predict_proba(Xenc)
      )

    finally:
      remove("lmc-sklearn2pmml.pmml")

  def test_sklearn2pmml_multiclass_multinomial(self):
    data = load_iris(as_frame=True)

    X = data.data
    y = data.target
    y.name = "Class"

    ref = LogisticRegression()
    ref.fit(X, y)

    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", ref)
    ])
    pipeline.fit(X, y)
    sklearn2pmml(pipeline, "lmc-sklearn2pmml.pmml", with_repr=True)

    try:
      # Import PMML
      model = PMMLLogisticRegression(pmml='lmc-sklearn2pmml.pmml')

      # Verify classification
      assert np.allclose(
        ref.predict_proba(X),
        model.predict_proba(X)
      )

    finally:
      remove("lmc-sklearn2pmml.pmml")

  def test_sklearn2pmml_multiclass_ovr(self):
    data = load_iris(as_frame=True)

    X = data.data
    y = data.target
    y.name = "Class"

    ref = LogisticRegression(
      multi_class='ovr'
    )
    ref.fit(X, y)

    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", ref)
    ])
    pipeline.fit(X, y)
    sklearn2pmml(pipeline, "lmc-sklearn2pmml.pmml", with_repr=True)

    try:
      # Import PMML
      model = PMMLLogisticRegression(pmml='lmc-sklearn2pmml.pmml')

      # Verify classification
      assert np.allclose(
        ref.predict_proba(X),
        model.predict_proba(X)
      )

    finally:
      remove("lmc-sklearn2pmml.pmml")

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == LogisticRegression()._more_tags()


class TestGeneralRegression(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLRidge(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain GeneralRegressionModel.'

  def test_invalid_classifier(self):
    with self.assertRaises(Exception) as cm:
      PMMLRidgeClassifier(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain GeneralRegressionModel.'

  def test_nonlinear_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLRidge(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="a" optype="continuous" dataType="double"/>
                </DataDictionary>
                <GeneralRegressionModel>
                  <MiningSchema>
                    <MiningField name="Class" usageType="target"/>
                  </MiningSchema>
                  <PPMatrix>
                    <PPCell value="1" predictorName="a" parameterName="p1"/>
                    <PPCell value="1" predictorName="a" parameterName="p1"/>
                  </PPMatrix>
                </GeneralRegressionModel>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model is not linear.'

  def test_multioutput_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLRidge(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="a" optype="continuous" dataType="double"/>
                </DataDictionary>
                <GeneralRegressionModel>
                  <MiningSchema>
                    <MiningField name="Class" usageType="target"/>
                  </MiningSchema>
                  <PPMatrix>
                    <PPCell value="1" predictorName="a" parameterName="p1"/>
                  </PPMatrix>
                  <ParamMatrix>
                    <PCell parameterName="p1" df="1" beta="1"/>
                    <PCell parameterName="p1" targetCategory="second" df="1" beta="1"/>
                  </ParamMatrix>
                </GeneralRegressionModel>
              </PMML>
              """))

    assert str(cm.exception) == 'This model does not support multiple outputs.'


class TestGeneralRegressionIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-glm.pmml')
    self.clf = PMMLRidge(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array([0.76984714, 0.62777417, 0.97269501, 0.92573904, 1.07729711, 0.84270291, 0.49553349, 0.71685506, 0.47088161, 0.97486390, 1.05130371, 0.61683889, 0.91572548, 0.77157660, 0.62100866, 0.58980751, 0.23293754, 1.05549000, 0.73136668, 0.91028562, 0.98442322, 0.76697277, 0.54041194, 0.66282497, 0.55121962, 0.50143919, 0.76523718, 0.14555227, 0.05832986, 0.33383867, 0.53914144, -0.11052323, 0.40016843, 0.22597578, 0.14323672, 0.51625628, 0.36130025, 0.39572621, 0.46020273, 0.52182059, -0.00768403, 0.26640930, 0.20815075, 0.38098647, 0.49802258, 0.56473838, 0.24103994, 0.26506002, 0.52001876, 0.14958276, 0.38839055, 0.46138168])
    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.4791710734180739
    assert np.allclose(ref, self.clf.score(Xte, yte == 'Yes'))


class TestRidgeIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xte = pd.get_dummies(Xte, prefix_sep='')
    del Xte['age(20,30]']
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-ridge.pmml')
    self.clf = PMMLRidge(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array([0.56707253, 0.44086932, 0.70106631, 0.63462966, 0.75552995, 0.60126409, 0.4352619 , 0.55362532, 0.40207959, 0.68526355, 0.77666758, 0.53249166, 0.61717879, 0.51593912, 0.49949509, 0.49068951, 0.26094857, 0.71970929, 0.57488419, 0.61499657, 0.6551319 , 0.59615382, 0.42850703, 0.52000645, 0.44016652, 0.42641415, 0.56069061, 0.21493887, 0.17195355, 0.33184511, 0.4237941 , 0.08433666, 0.34454511, 0.26253933, 0.23076609, 0.39833734, 0.35012744, 0.36532649, 0.42733187, 0.42595108, 0.18051046, 0.28151586, 0.25718191, 0.38083643, 0.43149017, 0.46942765, 0.29962233, 0.31491245, 0.49074276, 0.19720312, 0.36989965, 0.41818817])
    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.3286660932879891
    assert ref == self.clf.score(Xte, yte == 'Yes')

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == Ridge()._more_tags()


class TestRidgeClassifierIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-ridgec.pmml')
    self.clf = PMMLRidgeClassifier(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array(['Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','No','No','No','No','No','No','No','No','No','No','No','Yes','No','No','No','No','No','No','No','No','No','No','No','No','No'])
    assert np.all(ref == self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.9230769230769231
    assert np.allclose(ref, self.clf.score(Xte, yte))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == RidgeClassifier()._more_tags()


class TestLassoIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xte = pd.get_dummies(Xte, prefix_sep='')
    del Xte['age(20,30]']
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-lasso.pmml')
    self.clf = PMMLLasso(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array([0.54760797, 0.38252085, 0.64405119, 0.69267769, 0.75267467, 0.64973490, 0.55853793, 0.68095342, 0.43982239, 0.63910989, 0.66383460, 0.52450774, 0.83004930, 0.73954792, 0.62750939, 0.62808670, 0.35907729, 0.79067870, 0.56363984, 0.66214774, 0.76835019, 0.53462166, 0.50177534, 0.56021176, 0.58576734, 0.54497646, 0.71855174, 0.33393039, 0.30558116, 0.41401622, 0.58820140, 0.17572293, 0.47128396, 0.32379699, 0.31160441, 0.48678035, 0.45300624, 0.36171583, 0.55043818, 0.49585081, 0.50409265, 0.40823653, 0.21645944, 0.45602514, 0.34953902, 0.49245104, 0.37850364, 0.40465109, 0.42816803, 0.29311945, 0.48653454, 0.54348106])
    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.2878302689160125
    assert np.allclose(ref, self.clf.score(Xte, yte == 'Yes'))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == Lasso()._more_tags()


class TestElasticNetIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xte = pd.get_dummies(Xte, prefix_sep='')
    del Xte['age(20,30]']
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/linear-model-lasso.pmml')
    self.clf = PMMLElasticNet(pmml)

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array([0.54760797, 0.38252085, 0.64405119, 0.69267769, 0.75267467, 0.64973490, 0.55853793, 0.68095342, 0.43982239, 0.63910989, 0.66383460, 0.52450774, 0.83004930, 0.73954792, 0.62750939, 0.62808670, 0.35907729, 0.79067870, 0.56363984, 0.66214774, 0.76835019, 0.53462166, 0.50177534, 0.56021176, 0.58576734, 0.54497646, 0.71855174, 0.33393039, 0.30558116, 0.41401622, 0.58820140, 0.17572293, 0.47128396, 0.32379699, 0.31160441, 0.48678035, 0.45300624, 0.36171583, 0.55043818, 0.49585081, 0.50409265, 0.40823653, 0.21645944, 0.45602514, 0.34953902, 0.49245104, 0.37850364, 0.40465109, 0.42816803, 0.29311945, 0.48653454, 0.54348106])
    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.2878302689160125
    assert np.allclose(ref, self.clf.score(Xte, yte == 'Yes'))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    assert self.clf._more_tags() == ElasticNet()._more_tags()

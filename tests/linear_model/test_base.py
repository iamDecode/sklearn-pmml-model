from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.linear_model import PMMLLinearRegression, PMMLRidge
import pandas as pd
import numpy as np
from os import path
from io import StringIO


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


class TestLinearRegressionIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)

        pmml = path.join(BASE_DIR, '../models/linear-regression.pmml')
        self.clf = PMMLLinearRegression(pmml)

    def test_predict(self):
        Xte, _ = self.test
        ref = np.array([0.959047661, 0.306635481, 0.651029985, 0.954874880, 0.889268596, 0.874413539, 0.552911965, 0.793971929, 0.567604727, 0.694441984, 0.977588079, 1.020076443, 0.938209348, 0.741296266, 0.785681506, 0.783314305, 0.147203243, 0.953499858, 0.861694209, 0.818535888, 1.054586791, 0.508564304, 0.490740907, 0.692194962, 0.546339084, 0.584074930, 0.817451147, 0.007120341, -0.023103301, 0.354232979, 0.452602313, -0.232817829, 0.289612034, 0.241502904, 0.098388728, 0.413283786, 0.349024715, 0.315999598, 0.656973238, 0.525739661, 0.243258999, 0.128203855, 0.151826018, 0.357043960, 0.647876971, 0.405659892, 0.264334997, 0.280004394, 0.948749766, -0.028252457, 0.415301011, 0.509803923])
        assert np.allclose(ref, self.clf.predict(Xte))

    def test_score(self):
        Xte, yte = self.test
        ref = 0.409378064635437
        assert ref == self.clf.score(Xte, yte == 'Yes')


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
        assert ref == self.clf.score(Xte, yte == 'Yes')


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


class TestLassoIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xte = pd.get_dummies(Xte, prefix_sep='')
        del Xte['age(20,30]']
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)

        pmml = path.join(BASE_DIR, '../models/linear-model-lasso.pmml')
        self.clf = PMMLRidge(pmml)

    def test_predict(self):
        Xte, _ = self.test
        ref = np.array([0.54760797, 0.38252085, 0.64405119, 0.69267769, 0.75267467, 0.64973490, 0.55853793, 0.68095342, 0.43982239, 0.63910989, 0.66383460, 0.52450774, 0.83004930, 0.73954792, 0.62750939, 0.62808670, 0.35907729, 0.79067870, 0.56363984, 0.66214774, 0.76835019, 0.53462166, 0.50177534, 0.56021176, 0.58576734, 0.54497646, 0.71855174, 0.33393039, 0.30558116, 0.41401622, 0.58820140, 0.17572293, 0.47128396, 0.32379699, 0.31160441, 0.48678035, 0.45300624, 0.36171583, 0.55043818, 0.49585081, 0.50409265, 0.40823653, 0.21645944, 0.45602514, 0.34953902, 0.49245104, 0.37850364, 0.40465109, 0.42816803, 0.29311945, 0.48653454, 0.54348106])
        assert np.allclose(ref, self.clf.predict(Xte))

    def test_score(self):
        Xte, yte = self.test
        ref = 0.2878302689160125
        assert ref == self.clf.score(Xte, yte == 'Yes')


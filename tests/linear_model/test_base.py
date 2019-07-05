from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.linear_model import PMMLLinearRegression
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

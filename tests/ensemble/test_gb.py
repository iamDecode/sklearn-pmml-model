from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.ensemble import PMMLGradientBoostingClassifier, PMMLGradientBoostingRegressor
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from io import StringIO
import numpy as np
from os import path, remove
import pandas as pd


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestGradientBoosting(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLGradientBoostingClassifier(pmml=StringIO("""
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
      PMMLGradientBoostingClassifier(pmml=StringIO("""
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
      PMMLGradientBoostingClassifier(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model ensemble should use modelChain.'

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      pmml = path.join(BASE_DIR, '../models/gb-xgboost-iris.pmml')
      clf = PMMLGradientBoostingClassifier(pmml)
      clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    clf = PMMLGradientBoostingClassifier(path.join(BASE_DIR, '../models/gb-xgboost-iris.pmml'))
    assert clf._more_tags() == GradientBoostingClassifier()._more_tags()


class TestGradientBoostingRegression(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLGradientBoostingRegressor(pmml=StringIO("""
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
      PMMLGradientBoostingRegressor(pmml=StringIO("""
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
      PMMLGradientBoostingRegressor(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model ensemble should use sum.'

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      pmml = path.join(BASE_DIR, '../models/gb-gbm-cat-pima-regression.pmml')
      clf = PMMLGradientBoostingRegressor(pmml)
      clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_more_tags(self):
    clf = PMMLGradientBoostingRegressor(path.join(BASE_DIR, '../models/gb-gbm-cat-pima-regression.pmml'))
    assert clf._more_tags() == GradientBoostingRegressor()._more_tags()


class TestIrisGradientBoostingIntegration(TestCase):
  def setUp(self):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data)
    X.columns = np.array(['sepal_length', 'sepal_width', 'petal_length',
                          'petal_width'])
    y = pd.Series(np.array(np.array(['setosa', 'versicolor', 'virginica']))[iris.target])
    y.name = "Class"
    yreg = pd.Categorical(y).codes
    y2 = pd.Series(np.array(np.array(['setosa', 'other', 'other']))[iris.target])
    y2.name = "Class"
    self.test = X, y, y2, yreg
    self.ref = GradientBoostingClassifier(n_estimators=100, random_state=1).fit(X, y)
    self.ref_regression = GradientBoostingRegressor(n_estimators=100, random_state=1).fit(X, yreg)

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    pipeline.fit(self.test[0], self.test[1])

    sklearn2pmml(pipeline, "gb-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLGradientBoostingClassifier(pmml='gb-sklearn2pmml.pmml')

      # Verify classification
      Xte, yte, _, _ = self.test

      assert np.array_equal(
        self.ref.score(Xte, yte),
        model.score(Xte, yte)
      )

      assert np.allclose(
        self.ref.predict_proba(Xte),
        model.predict_proba(Xte)
      )

    finally:
      remove("gb-sklearn2pmml.pmml")

  def test_sklearn2pmml_binary(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    Xte, _, yte, _ = self.test
    pipeline.fit(Xte, yte)

    sklearn2pmml(pipeline, "gb-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLGradientBoostingClassifier(pmml='gb-sklearn2pmml.pmml')

      # Verify classification
      assert np.array_equal(
        self.ref.score(Xte, yte),
        model.score(Xte, yte)
      )

      assert np.allclose(
        self.ref.predict_proba(Xte),
        model.predict_proba(Xte)
      )

    finally:
      remove("gb-sklearn2pmml.pmml")

  def test_sklearn2pmml_regression(self):
    Xte, _, _, yte = self.test

    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref_regression)
    ])

    pipeline.fit(Xte, yte)
    sklearn2pmml(pipeline, "gb-sklearn2pmml-regression.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLGradientBoostingRegressor(pmml='gb-sklearn2pmml-regression.pmml')

      assert np.array_equal(
        self.ref_regression.score(Xte, yte),
        model.score(Xte, yte)
      )

      assert np.allclose(
        self.ref_regression.predict(Xte),
        model.predict(Xte)
      )

    finally:
      remove("gb-sklearn2pmml-regression.pmml")

  def test_R_xgboost(self):
    pmml = path.join(BASE_DIR, '../models/gb-xgboost-iris.pmml')
    clf = PMMLGradientBoostingClassifier(pmml)

    # Verify classification
    Xte, yte, _, _ = self.test

    ref = 0.9933333333333333
    assert ref == clf.score(Xte, yte)

    ref = [
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9757426417447441, 0.0128682851648328, 0.0113890730904230],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.9760594257941966, 0.0128724629749537, 0.0110681112308498],
      [0.0106166153088319, 0.9785400205188246, 0.0108433641723436],
      [0.0106166153088319, 0.9785400205188246, 0.0108433641723436],
      [0.0101155963333861, 0.9323607906747136, 0.0575236129919005],
      [0.0119572590868193, 0.9754718678170027, 0.0125708730961781],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0159323759619904, 0.9288613767350993, 0.0552062473029103],
      [0.0124389433616677, 0.9744837805406320, 0.0130772760977003],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0140073718554280, 0.9712664360693611, 0.0147261920752109],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0141667403550577, 0.9709395207122262, 0.0148937389327162],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106166153088319, 0.9785400205188246, 0.0108433641723436],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0141667403550577, 0.9709395207122262, 0.0148937389327162],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0345623929310359, 0.4340977426739484, 0.5313398643950156],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0100985567108226, 0.9307902380901225, 0.0591112051990549],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0104458453096068, 0.9628000437291715, 0.0267541109612217],
      [0.0367540158081575, 0.7002838055528258, 0.2629621786390166],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0498135826560765, 0.4929875804501168, 0.4571988368938067],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0159323759619904, 0.9288613767350993, 0.0552062473029103],
      [0.0106166153088319, 0.9785400205188246, 0.0108433641723436],
      [0.0119572590868193, 0.9754718678170027, 0.0125708730961781],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0140073718554280, 0.9712664360693611, 0.0147261920752109],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0106132395880432, 0.9782288782391356, 0.0111578821728212],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0164145902620253, 0.0255525167427301, 0.9580328929952446],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0435806598526578, 0.2599986718108350, 0.6964206683365073],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0142345906453859, 0.0312354373543125, 0.9545299720003015],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0184930026637306, 0.0305865778733861, 0.9509204194628834],
      [0.0164145902620253, 0.0255525167427301, 0.9580328929952446],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0510645228555055, 0.2463746443944004, 0.7025608327500941],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0184930026637306, 0.0305865778733861, 0.9509204194628834],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0125007199528260, 0.0186054175395307, 0.9688938625076433],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0299366378213490, 0.0965402806010371, 0.8735230815776139],
      [0.0120440726761590, 0.0544554457865047, 0.9335004815373361],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0464351273705071, 0.2443049639688132, 0.7092599086606799],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0363615404352432, 0.1786334851333196, 0.7850049744314372],
      [0.0252911348271332, 0.0660520596210707, 0.9086568055517961],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0250153077815250, 0.2450614165877344, 0.7299232756307406],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0142478270786007, 0.0303346044639386, 0.9554175684574607],
      [0.0164145902620253, 0.0255525167427301, 0.9580328929952446],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0108613904938508, 0.0140656636934379, 0.9750729458127113],
      [0.0125007199528260, 0.0186054175395307, 0.9688938625076433],
      [0.0108777095343255, 0.0125843153938132, 0.9765379750718612],
      [0.0108567094134873, 0.0144905851493065, 0.9746527054372062],
      [0.0162658935674777, 0.0343798537311987, 0.9493542527013236],
    ]
    assert np.allclose(
      ref,
      clf.predict_proba(Xte)
    )


class TestCategoricalPimaGradientBoostingIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats)
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/gb-gbm-cat-pima.pmml')
    self.clf = PMMLGradientBoostingClassifier(pmml)

  def test_predict_proba(self):
    Xte, yte = self.test

    ref = np.array([
      [0.0450486288734726, 0.9549513711265273],
      [0.5885160147778145, 0.4114839852221857],
      [0.0009551290028415, 0.9990448709971584],
      [0.0002422991756940, 0.9997577008243060],
      [0.0007366887385038, 0.9992633112614963],
      [0.0004013672347579, 0.9995986327652422],
      [0.0690449928421312, 0.9309550071578688],
      [0.0016227716534959, 0.9983772283465041],
      [0.1293482327095098, 0.8706517672904902],
      [0.0000087074095761, 0.9999912925904240],
      [0.0000791716298258, 0.9999208283701742],
      [0.0024433990519477, 0.9975566009480523],
      [0.0201993339302258, 0.9798006660697742],
      [0.0134746789867266, 0.9865253210132735],
      [0.2373888296249749, 0.7626111703750250],
      [0.0397951119541024, 0.9602048880458975],
      [0.6048657697720786, 0.3951342302279213],
      [0.0783995167388520, 0.9216004832611480],
      [0.0770031400315019, 0.9229968599684981],
      [0.0039225633807996, 0.9960774366192003],
      [0.0870982756252495, 0.9129017243747506],
      [0.0518153619780974, 0.9481846380219027],
      [0.0529373167466456, 0.9470626832533544],
      [0.1465303709628746, 0.8534696290371255],
      [0.2948688765091694, 0.7051311234908306],
      [0.0093530273605733, 0.9906469726394268],
      [0.8000454863890519, 0.1999545136109481],
      [0.9840595476042330, 0.0159404523957671],
      [0.9994934828677855, 0.0005065171322145],
      [0.9332373260900556, 0.0667626739099444],
      [0.8301575628390390, 0.1698424371609611],
      [0.9994014211291834, 0.0005985788708167],
      [0.9964913762897571, 0.0035086237102429],
      [0.9877457479606355, 0.0122542520393643],
      [0.9866554945920915, 0.0133445054079084],
      [0.9933345920713418, 0.0066654079286583],
      [0.9987899464870794, 0.0012100535129206],
      [0.9815251490646326, 0.0184748509353674],
      [0.8197681630417475, 0.1802318369582524],
      [0.6525706661194860, 0.3474293338805139],
      [0.8825708419171311, 0.1174291580828689],
      [0.9840595476042330, 0.0159404523957671],
      [0.8106506748060572, 0.1893493251939427],
      [0.9990627397651777, 0.0009372602348224],
      [0.8763278695225744, 0.1236721304774257],
      [0.7824982271907117, 0.2175017728092884],
      [0.9992088489858751, 0.0007911510141249],
      [0.9994453225672126, 0.0005546774327873],
      [0.5301562607268657, 0.4698437392731343],
      [0.9877457479606355, 0.0122542520393643],
      [0.9698651982991124, 0.0301348017008876],
      [0.7011874523008232, 0.2988125476991768],
    ])

    assert np.allclose(ref, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.9615384615384616
    assert ref == self.clf.score(Xte, yte)


class TestCategoricalPimaGradientBoostingRegressionIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats)
    Xte = df.iloc[:, 1:]
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)

    pmml = path.join(BASE_DIR, '../models/gb-gbm-cat-pima-regression.pmml')
    self.clf = PMMLGradientBoostingRegressor(pmml)

  def test_predict_proba(self):
    Xte, yte = self.test

    ref = np.array([8.7963558386421514, 4.2301487026115723, 10.3074130259255288, 11.0860042614012890, 10.8390882310787244, 11.6744276206624455, 6.8608053925382197, 9.0998439359242642, 6.1702581355388837, 12.9114410951622993, 12.0474364608163835, 9.4157358501361372, 8.9477742716855513, 8.6419978545111871, 6.7908113300154973, 7.8564460175848332, 2.5803039557291312, 7.4870246797282878, 8.8266231982794476, 10.0496192384298837, 8.4405425349964833, 7.1471731945422761, 6.7398521259839423, 6.7249846985687745, 6.0319108950989166, 8.4379837918667207, 4.8378531067712780,-0.1407388954840041,-0.4792882429478054, 1.4349186285165105, 4.3101553536690291,-0.7930196993237724, 0.4815607255042949, 1.4538897420629540, 1.0425325111888282, 2.0425202398853499, 0.3823644412175868, 1.7699741003671532, 4.4468787718050429, 4.4858266945650191, 2.6713073344830360,-0.1407388954840041, 3.3409551722776252, 0.1230805829857022, 2.7432385943292585, 4.4890530381860367, 0.1609319868079373, 0.0683164397100340, 4.9337076961523039, 1.6038396339595518, 2.9159757524348273, 4.9144114866512734])

    assert np.allclose(ref, self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.6847123751907591
    assert ref == self.clf.score(Xte, (yte == "Yes")*10)

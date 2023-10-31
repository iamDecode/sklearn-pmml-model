from unittest import TestCase
from io import StringIO
import sklearn_pmml_model
from sklearn_pmml_model.auto_detect import auto_detect_estimator
from sklearn_pmml_model.tree import PMMLTreeClassifier, PMMLTreeRegressor
from sklearn_pmml_model.ensemble import PMMLForestClassifier, PMMLForestRegressor, PMMLGradientBoostingClassifier, \
  PMMLGradientBoostingRegressor
from sklearn_pmml_model.neural_network import PMMLMLPClassifier, PMMLMLPRegressor
from sklearn_pmml_model.svm import PMMLSVC, PMMLSVR
from sklearn_pmml_model.naive_bayes import PMMLGaussianNB
from sklearn_pmml_model.linear_model import PMMLLogisticRegression, PMMLLinearRegression, PMMLRidgeClassifier, PMMLRidge
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier, PMMLKNeighborsRegressor

from os import path

BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestAutoDetect(TestCase):
  def test_auto_detect_unsupported_classifier(self):
    with self.assertRaises(Exception) as cm:
      auto_detect_estimator(StringIO("""
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

    assert str(cm.exception) == 'Unsupported PMML classifier.'

  def test_auto_detect_unsupported_regressor(self):
    with self.assertRaises(Exception) as cm:
      auto_detect_estimator(StringIO("""
        <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
          <DataDictionary>
            <DataField name="Class" optype="continuous" dataType="float"/>
          </DataDictionary>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
        </PMML>
        """))

    assert str(cm.exception) == 'Unsupported PMML regressor.'

  def test_auto_detect_invalid_classifier_segmentation(self):
    with self.assertRaises(Exception) as cm:
      auto_detect_estimator(StringIO("""
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
          <Segmentation>
            <TreeModel />
            <SupportVectorMachine />
          </Segmentation>
        </PMML>
        """))

    assert str(cm.exception) == 'Unsupported PMML classifier: invalid segmentation.'

  def test_auto_detect_invalid_regressor_segmentation(self):
    with self.assertRaises(Exception) as cm:
      auto_detect_estimator(StringIO("""
        <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
          <DataDictionary>
            <DataField name="Class" optype="continuous" dataType="float"/>
          </DataDictionary>
          <MiningSchema>
            <MiningField name="Class" usageType="target"/>
          </MiningSchema>
          <Segmentation>
            <TreeModel />
            <SupportVectorMachine />
          </Segmentation>
        </PMML>
        """))

    assert str(cm.exception) == 'Unsupported PMML regressor: invalid segmentation.'

  def test_auto_detect_tree_classifier(self):
    pmml = path.join(BASE_DIR, '../models/tree-iris.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLTreeClassifier)

  def test_auto_detect_tree_regressor(self):
    pmml = path.join(BASE_DIR, '../models/tree-cat-pima-regression.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLTreeRegressor)

  def test_auto_detect_forest_classifier(self):
    pmml = path.join(BASE_DIR, '../models/rf-cat-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLForestClassifier)

  def test_auto_detect_forest_regressor(self):
    pmml = path.join(BASE_DIR, '../models/rf-cat-pima-regression.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLForestRegressor)

  def test_auto_detect_gradient_boosting_classifier(self):
    pmml = path.join(BASE_DIR, '../models/gb-xgboost-iris.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLGradientBoostingClassifier)

  def test_auto_detect_gradient_boosting_regressor(self):
    pmml = path.join(BASE_DIR, '../models/gb-gbm-cat-pima-regression.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLGradientBoostingRegressor)

  def test_auto_detect_neural_network_classifier(self):
    pmml = path.join(BASE_DIR, '../models/nn-iris.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLMLPClassifier)

  def test_auto_detect_neural_network_regressor(self):
    pmml = path.join(BASE_DIR, '../models/nn-pima-regression.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLMLPRegressor)

  def test_auto_detect_svm_classifier(self):
    pmml = path.join(BASE_DIR, '../models/svc-cat-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLSVC)

  def test_auto_detect_svm_regressor(self):
    pmml = path.join(BASE_DIR, '../models/svr-cat-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLSVR)

  def test_auto_detect_naive_bayes_classifier(self):
    pmml = path.join(BASE_DIR, '../models/nb-cat-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLGaussianNB)

  def test_auto_detect_linear_regression_classifier(self):
    pmml = path.join(BASE_DIR, '../models/linear-model-lmc.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLLogisticRegression)

  def test_auto_detect_linear_regression_regressor(self):
    pmml = path.join(BASE_DIR, '../models/linear-model-lm.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLLinearRegression)

  def test_auto_detect_general_regression_classifier(self):
    pmml = path.join(BASE_DIR, '../models/linear-model-ridgec.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLRidgeClassifier)

  def test_auto_detect_general_regression_regressor(self):
    pmml = path.join(BASE_DIR, '../models/linear-model-glm.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLRidge)

  def test_auto_detect_knn_classifier(self):
    pmml = path.join(BASE_DIR, '../models/knn-clf-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLKNeighborsClassifier)

  def test_auto_detect_knn_regressor(self):
    pmml = path.join(BASE_DIR, '../models/knn-reg-pima.pmml')
    assert isinstance(auto_detect_estimator(pmml=pmml), PMMLKNeighborsRegressor)

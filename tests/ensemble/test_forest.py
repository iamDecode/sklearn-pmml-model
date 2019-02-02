from unittest import TestCase
import sklearn_pmml_model
from sklearn_pmml_model.ensemble import PMMLForestClassifier
from io import StringIO
import numpy as np
from os import path
import pandas as pd
import warnings


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

    assert str(cm.exception) == 'PMML model ensemble should use majority vote.'

  def test_non_true_segment(self):
    with self.assertRaises(Exception) as cm, warnings.catch_warnings(record=True) as w:
      clf = PMMLForestClassifier(pmml=StringIO("""
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
    assert len(w) == 1

  def test_non_binary_tree(self):
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
          <Segmentation multipleModelMethod="majorityVote">
            <Segment>
              <True/>
              <TreeModel splitCharacteristic="multiSplit" />
            </Segment>
          </Segmentation>
        </MiningModel>
      </PMML>
      """))

    assert str(cm.exception) == 'Sklearn only supports binary tree models.'

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      clf = PMMLForestClassifier(pmml=path.join(BASE_DIR, '../models/categorical-rf.pmml'))
      clf.fit(np.array([[]]),np.array([]))

    assert str(cm.exception) == 'Not supported.'


class TestCategoricalPimaForestIntegration(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    cats = np.unique(df['age'])
    df['age'] = pd.Categorical(df['age'], categories=cats)
    df['age'] = df['age'].cat.codes
    Xte = df.iloc[:,1:]
    yte = df.iloc[:,0]
    self.test = (Xte, yte)

    self.clf = PMMLForestClassifier(pmml=path.join(BASE_DIR, '../models/categorical-rf.pmml'))

  def test_predict_proba(self):
    Xte, _ = self.test
    reference = np.array([
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
    assert np.array_equal(reference, self.clf.predict_proba(Xte))

  def test_score(self):
    Xte, yte = self.test
    reference = 0.7884615384615384
    assert reference == self.clf.score(Xte, yte)

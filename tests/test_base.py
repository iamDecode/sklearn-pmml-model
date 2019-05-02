from unittest import TestCase
from sklearn_pmml_model.base import PMMLBaseEstimator, get_type
from sklearn_pmml_model.datatypes import Category
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from io import StringIO
from collections import namedtuple

# Parameters
pair = [0, 1]

# Load data
iris = load_iris()

# We only take the two corresponding features
X = pd.DataFrame(iris.data[:, pair])
X.columns = np.array(iris.feature_names)[pair]
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"


class TestBase(TestCase):
  def test_evaluate_feature_mapping(self):
    clf = PMMLBaseEstimator(pmml=StringIO("""
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="Class" optype="categorical" dataType="string">
          <Value value="setosa"/>
          <Value value="versicolor"/>
          <Value value="virginica"/>
        </DataField>
        <DataField name="sepal length (cm)" optype="continuous" dataType="float"/>
        <DataField name="sepal width (cm)" optype="continuous" dataType="float"/>
      </DataDictionary>
      <TransformationDictionary>
        <DerivedField name="integer(sepal length (cm))" optype="continuous" dataType="integer">
          <FieldRef field="sepal length (cm)"/>
        </DerivedField>
        <DerivedField name="double(sepal width (cm))" optype="continuous" dataType="double">
          <FieldRef field="sepal width (cm)"/>
        </DerivedField>
      </TransformationDictionary>
      <MiningSchema>
        <MiningField name="Class" usageType="target"/>
      </MiningSchema>
    </PMML>
    """))

    Result = namedtuple('Result', 'column type')
    category = Category(str,
                        categories=['setosa', 'versicolor', 'virginica'],
                        ordered=False)
    tests = {
      'sepal length (cm)':          Result(column=0, type=float),
      'sepal width (cm)':           Result(column=1, type=float),
      'integer(sepal length (cm))': Result(column=0, type=int),
      'double(sepal width (cm))':   Result(column=1, type=float),
      'Class':                      Result(column=None, type=category),
    }

    for i in range(0, X.shape[0]):
      for feature, result in tests.items():
        column, data_type = clf.field_mapping[feature]
        assert column == result.column
        assert data_type == result.type

  def test_target_field(self):
    clf = PMMLBaseEstimator(pmml=StringIO("""
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>
    """))
    assert clf.target_field is None

    clf = PMMLBaseEstimator(pmml=StringIO("""
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
    assert clf.target_field.get('name') == 'Class'
    assert clf.target_field.get('optype') == 'categorical'
    assert clf.target_field.get('dataType') == 'string'

  def test_parse_type_value_continuous(self):
    template = """
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="test" optype="{}" dataType="{}"/>
      </DataDictionary>
    </PMML>"""

    values = ["1234", 1234, 12.34, True]
    types = [
      ['continuous', 'integer', int],
      ['continuous', 'float', float],
      ['continuous', 'double', float],
    ]

    for value in values:
      for field_type in types:
        op_type, pmml_type, data_type = field_type
        clf = PMMLBaseEstimator(StringIO(template.format(op_type, pmml_type)))

        data_dictionary = clf.root.find("DataDictionary")
        data_field = data_dictionary.find("DataField")
        result = get_type(data_field)(value)

        assert isinstance(result, data_type)

  def test_parse_type_value_continuous_boolean(self):
    template = """
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="test" optype="{}" dataType="{}"/>
      </DataDictionary>
    </PMML>"""

    tests = {
      "1": True,
      "True": True,
      "YES": True,
      1: True,
      True: True,
      "0": False,
      "False": False,
      0: False
    }

    for value, expected in tests.items():
      pmml = StringIO(template.format('continuous', 'boolean'))
      clf = PMMLBaseEstimator(pmml)

      data_dictionary = clf.root.find("DataDictionary")
      data_field = data_dictionary.find("DataField")
      result = get_type(data_field)(value)

      assert isinstance(result, bool)
      assert result == expected

  def test_get_type_exception(self):
    template = """
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="test" optype="{}" dataType="{}"/>
      </DataDictionary>
    </PMML>"""

    # Test invalid data type
    pmml = StringIO(template.format("continuous", "does_not_exist"))
    clf = PMMLBaseEstimator(pmml)
    data_dictionary = clf.root.find("DataDictionary")
    data_field = data_dictionary.find("DataField")

    with self.assertRaises(Exception) as cm:
      get_type(data_field)
    assert str(cm.exception) == "Unsupported data type."

    # Test invalid operation type
    pmml = StringIO(template.format("does_not_exist", "string"))
    clf = PMMLBaseEstimator(pmml)
    data_dictionary = clf.root.find("DataDictionary")
    data_field = data_dictionary.find("DataField")

    with self.assertRaises(Exception) as cm:
      get_type(data_field)
    assert str(cm.exception) == "Unsupported operation type."

  def test_get_type_categorical(self):
      template = """
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Class" optype="categorical" dataType="string">
            <Value value="setosa"/>
            <Value value="versicolor"/>
            <Value value="virginica"/>
          </DataField>
        </DataDictionary>
      </PMML>"""

      clf = PMMLBaseEstimator(pmml=StringIO(template))
      data_dictionary = clf.root.find("DataDictionary")
      data_field = data_dictionary.find("DataField")
      data_type: Category = get_type(data_field)

      assert data_type.categories == ['setosa', 'versicolor', 'virginica']
      assert not data_type.ordered

  def test_get_type_ordinal(self):
    template = """
      <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
        <DataDictionary>
          <DataField name="Volume" optype="ordinal" dataType="string">
            <Value value="loud"/>
            <Value value="louder"/>
            <Value value="loudest"/>
          </DataField>
        </DataDictionary>
      </PMML>"""

    clf = PMMLBaseEstimator(pmml=StringIO(template))
    data_dictionary = clf.root.find("DataDictionary")
    data_field = data_dictionary.find("DataField")
    data_type: Category = get_type(data_field)

    assert data_type.categories == ['loud', 'louder', 'loudest']
    assert data_type.ordered

  def test_fit_exception(self):
    pmml = StringIO('<PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>')
    clf = PMMLBaseEstimator(pmml)

    with self.assertRaises(Exception) as cm:
      clf.fit(X, y)

    assert str(cm.exception) == "Not supported."

from unittest import TestCase
from sklearn_pmml_model.base import PMMLBaseEstimator
from sklearn_pmml_model.datatypes import Category, Boolean, Interval
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
    tests = {
      'sepal length (cm)':          Result(column=0, type=float),
      'sepal width (cm)':           Result(column=1, type=float),
      'integer(sepal length (cm))': Result(column=0, type=int),
      'double(sepal width (cm))':   Result(column=1, type=float),
      'Class':                      Result(column=None, type=Category),
    }

    for i in range(0, X.shape[0]):
      for feature, result in tests.items():
        column, mapping = clf.field_mapping[feature]
        assert column == result.column
        mapped_value = mapping(X.iloc[i][column]) if column is not None else mapping(y.iloc[i])
        assert type(mapped_value) == result.type

        if result.type == Category:
          assert mapped_value.value == y.iloc[i]
          assert mapped_value.categories == ["setosa", "versicolor", "virginica"]
        else:
          assert mapped_value == result.type(X.iloc[i][column])

  def test_target_field(self):
    clf = PMMLBaseEstimator(pmml=StringIO("""
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>
    """))
    assert clf.target_field == None

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
      for type in types:
        optype, pmml_type, data_type = type
        clf = PMMLBaseEstimator(pmml=StringIO(template.format(optype, pmml_type)))

        data_dictionary = clf.find(clf.root, "DataDictionary")
        data_field = clf.find(data_dictionary, "DataField")
        result = clf.parse_type(value, data_field)

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
      clf = PMMLBaseEstimator(pmml=StringIO(template.format('continuous', 'boolean')))

      data_dictionary = clf.find(clf.root, "DataDictionary")
      data_field = clf.find(data_dictionary, "DataField")
      result = clf.parse_type(value, data_field)

      assert isinstance(result, Boolean)
      assert result == expected

  def test_parse_type_value_exception(self):
    template = """
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="test" optype="{}" dataType="{}"/>
      </DataDictionary>
    </PMML>"""

    # Test invalid data type
    clf = PMMLBaseEstimator(pmml=StringIO(template.format("continuous", "does_not_exist")))
    data_dictionary = clf.find(clf.root, "DataDictionary")
    data_field = clf.find(data_dictionary, "DataField")

    with self.assertRaises(Exception) as cm: clf.parse_type("test", data_field)
    assert str(cm.exception) == "Unsupported data type."

    # Test invalid operation type
    clf = PMMLBaseEstimator(pmml=StringIO(template.format("does_not_exist", "string")))
    data_dictionary = clf.find(clf.root, "DataDictionary")
    data_field = clf.find(data_dictionary, "DataField")

    with self.assertRaises(Exception) as cm: clf.parse_type("test", data_field)
    assert str(cm.exception) == "Unsupported operation type."

  def test_parse_type_value_categorical(self):
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
      data_dictionary = clf.find(clf.root, "DataDictionary")
      data_field = clf.find(data_dictionary, "DataField")

      with self.assertRaises(Exception) as cm: clf.parse_type("not_in_category", data_field)
      assert str(cm.exception) == "Value does not match any category."
      assert clf.parse_type("setosa", data_field) == "setosa"
      assert clf.parse_type("versicolor", data_field) == "versicolor"
      assert clf.parse_type("virginica", data_field) == "virginica"
      assert isinstance(clf.parse_type("virginica", data_field), Category)
      assert isinstance(clf.parse_type("virginica", data_field, force_native=True), str)

  def test_parse_type_value_ordinal(self):
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
    data_dictionary = clf.find(clf.root, "DataDictionary")
    data_field = clf.find(data_dictionary, "DataField")

    with self.assertRaises(Exception)as cm: clf.parse_type("not_in_category", data_field)
    assert str(cm.exception) == "Value does not match any category."
    assert clf.parse_type("loud", data_field) == "loud"
    assert clf.parse_type("louder", data_field) == "louder"
    assert clf.parse_type("loudest", data_field) == "loudest"
    assert isinstance(clf.parse_type("loudest", data_field), Category)
    assert isinstance(clf.parse_type("loudest", data_field, force_native=True), str)

    assert clf.parse_type("loud", data_field) < clf.parse_type("louder", data_field)
    assert clf.parse_type("louder", data_field) < clf.parse_type("loudest", data_field)

  def test_parse_type_interval(self):
    template = """
    <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
      <DataDictionary>
        <DataField name="test" optype="ordinal" dataType="float">
          <Interval closure="openOpen" rightMargin="1"/>
          <Interval closure="openClosed" leftMargin="1" rightMargin="1.5"/>
          <Interval closure="openOpen" leftMargin="1.5" rightMargin="2.5"/>
          <Interval closure="closedOpen" leftMargin="2.5" rightMargin="3.5"/>
          <Interval closure="closedClosed" leftMargin="3.5" />
        </DataField>
      </DataDictionary>
    </PMML>"""

    clf = PMMLBaseEstimator(pmml=StringIO(template))
    data_dictionary = clf.find(clf.root, "DataDictionary")
    data_field = clf.find(data_dictionary, "DataField")

    assert clf.parse_type(-1, data_field) == Interval(-1, rightMargin=1, closure='openOpen')
    with self.assertRaises(Exception): clf.parse_type(1, data_field)
    assert clf.parse_type(2, data_field) == Interval(2, leftMargin=1.5, rightMargin=2.5, closure='openOpen')
    assert clf.parse_type(2.5, data_field) == Interval(2.5, leftMargin=2.5, rightMargin=3.5, closure='closedOpen')
    assert clf.parse_type(3.5, data_field) == Interval(3.5, leftMargin=3.5, closure='closedClosed')
    assert clf.parse_type(3.5, data_field, force_native=True) == 3.5

  def test_fit_exception(self):
    clf = PMMLBaseEstimator(pmml=StringIO('<PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3"/>'))
    with self.assertRaises(Exception) as cm:
      clf.fit(X, y)

    assert str(cm.exception) == "Not supported."

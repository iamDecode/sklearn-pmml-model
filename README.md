# sklearn-pmml-model

[![CircleCI](https://circleci.com/gh/iamDecode/sklearn-pmml-model.svg?style=shield)](https://circleci.com/gh/iamDecode/sklearn-pmml-model)

A library to parse PMML models into Scikit-learn estimators.

## Installing

The easiest way is to use pip:

```
$ pip install sklearn-pmml-model
```

## Status
This library is very alpha, and currently only supports a small part of the [specification](http://dmg.org/pmml/v4-3/GeneralStructure.html):
- DataDictionary
  - DataField (continuous, categorical, ordinal)
    - Value
    - Interval
- TransformationDictionary
  - DerivedField
- TreeModel
  - Node
    - SimplePredicate (eq, neq, lt, le, gt, ge)
    
For a minimum viable beta we like to at least add `SimpleSetPredicate` and support for all dataTypes (date, time, dateTime etc).

## Example
A minimal working example is shown below:

```python
# Prepare data
iris = load_iris()

# We only take the two corresponding features
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

df = pd.concat([Xte, yte], axis=1)

clf = PMMLTreeClassifier(pmml="../models/DecisionTreeIris.pmml")
clf.predict(Xte)
clf.score(Xte, yte)
```

## Running the tests

Tests can be run using Py.test. First install dev dependencies:

```
$ pip install -r requirements-dev.txt
```

Then execute tests by running:
```
$ pytest tests/
```

## Contributing

Feel free to make a contribution. Please read [CONTRIBUTING.md]() for details on the code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE.md](LICENSE.md) file for details.

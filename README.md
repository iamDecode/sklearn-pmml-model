<img src="https://user-images.githubusercontent.com/1223300/41346080-c2c910a0-6f05-11e8-89e9-71a72bb9543f.png" width="300">

# sklearn-pmml-model

[![PyPI version](https://badge.fury.io/py/sklearn-pmml-model.svg)](https://badge.fury.io/py/sklearn-pmml-model)
[![CircleCI](https://circleci.com/gh/iamDecode/sklearn-pmml-model.svg?style=shield)](https://circleci.com/gh/iamDecode/sklearn-pmml-model)
[![codecov](https://codecov.io/gh/iamDecode/sklearn-pmml-model/branch/master/graph/badge.svg?token=CGbbgziGwn)](https://codecov.io/gh/iamDecode/sklearn-pmml-model)

A library to parse PMML models into Scikit-learn estimators.

## Installation

The easiest way is to use pip:

```
$ pip install sklearn-pmml-model
```

## Status
This library is very alpha, and currently only supports a limited number of models. The library currently supports the following models:
- [Decision Trees](sklearn_pmml_model/tree) (`sklearn_pmml_model.tree.PMMLTreeClassifier`)
- [Random Forests](sklearn_pmml_model/ensemble) (`sklearn_pmml_model.ensemble.PMMLForestClassifier`)
- [Linear Regression](sklearn_pmml_model/linear_model) (`sklearn_pmml_model.linear_model.PMMLLinearRegression`)
- [Ridge](sklearn_pmml_model/linear_model) (`sklearn_pmml_model.linear_model.PMMLRidge`)
- [Lasso](sklearn_pmml_model/linear_model) (`sklearn_pmml_model.linear_model.PMMLLasso`)
- [ElasticNet](sklearn_pmml_model/linear_model) (`sklearn_pmml_model.linear_model.PMMLElasticNet`)

A small part of the [specification](http://dmg.org/pmml/v4-3/GeneralStructure.html) is covered:
- DataDictionary
  - DataField (continuous, categorical, ordinal)
    - Value
    - Interval
- TransformationDictionary
  - DerivedField
- TreeModel
  - SimplePredicate
  - SimpleSetPredicate
- Segmentation ('majorityVote' only, for Random Forests)
- Regression
  - RegressionTable
    - NumericPredictor
    - CategoricalPredictor
- GeneralRegressionModel (only linear models)
  - PPMatrix
    - PPCell
  - ParamMatrix
    - PCell
  
## Example
A minimal working example is shown below:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.ensemble import PMMLForestClassifier

# Prepare data
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

clf = PMMLForestClassifier(pmml="models/randomForest.pmml")
clf.predict(Xte)
clf.score(Xte, yte)
```

More examples can be found in the subsequent packages: [tree](sklearn_pmml_model/tree), [ensemble](sklearn_pmml_model/ensemble).
## Development

### Prerequisites

Tests can be run using Py.test. Grab a local copy of the source:

```
$ git clone http://github.com/iamDecode/sklearn-pmml-model
```

create a virtual environment:
```
$ python3 -m venv venv
```

And install the dependencies:

```
$ pip install -r requirements.txt
```

### Testing

You can execute tests with py.test by running:
```
$ python setup.py pytest
```

## Contributing

Feel free to make a contribution. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

<img src="https://user-images.githubusercontent.com/1223300/41346080-c2c910a0-6f05-11e8-89e9-71a72bb9543f.png" width="300">

# sklearn-pmml-model

[![PyPI version](https://badge.fury.io/py/sklearn-pmml-model.svg)](https://badge.fury.io/py/sklearn-pmml-model)
[![CircleCI](https://circleci.com/gh/iamDecode/sklearn-pmml-model.svg?style=shield)](https://circleci.com/gh/iamDecode/sklearn-pmml-model)
[![codecov](https://codecov.io/gh/iamDecode/sklearn-pmml-model/branch/master/graph/badge.svg?token=CGbbgziGwn)](https://codecov.io/gh/iamDecode/sklearn-pmml-model)
[![ReadTheDocs](https://readthedocs.org/projects/sklearn-pmml-model/badge/?version=latest&style=flat)](https://sklearn-pmml-model.readthedocs.io/en/latest/)

A library to parse PMML models into Scikit-learn estimators.

## Installation

The easiest way is to use pip:

```
$ pip install sklearn-pmml-model
```

## Status
This library is in beta, and currently not all models are supported. The library currently does support the following models:

| Model                                                  | Classification | Regression | Categorical features |
|--------------------------------------------------------|----------------|------------|----------------------|
| [Decision Trees](sklearn_pmml_model/tree)              | ✅             | ✅         | ✅<sup>1</sup>        |
| [Random Forests](sklearn_pmml_model/ensemble)          | ✅             | ✅         | ✅<sup>1</sup>        |
| [Gradient Boosting](sklearn_pmml_model/ensemble)       | ✅             | ✅         | ✅<sup>1</sup>        |
| [Linear Regression](sklearn_pmml_model/linear_model)   | ✅             | ✅         | ✅<sup>3</sup>        |
| [Ridge](sklearn_pmml_model/linear_model)               | ✅<sup>2</sup> | ✅         | ✅<sup>3</sup>        |
| [Lasso](sklearn_pmml_model/linear_model)               | ✅<sup>2</sup> | ✅         | ✅<sup>3</sup>        |
| [ElasticNet](sklearn_pmml_model/linear_model)          | ✅<sup>2</sup> | ✅         | ✅                    |
| [Gaussian Naive Bayes](sklearn_pmml_model/naive_bayes) | ✅             |            |                      |

<sup>1</sup> Categorical feature support using slightly modified internals, based on [scikit-learn#12866](https://github.com/scikit-learn/scikit-learn/pull/12866).

<sup>2</sup> These models differ only in training characteristics, the resulting model is of the same form. Classification is supported using `PMMLLogisticRegression` for regression models and `PMMLRidgeClassifier` for general regression models.

<sup>3</sup> By one-hot encoding categorical features automatically.

---

The following part of the [specification](http://dmg.org/pmml/v4-3/GeneralStructure.html) is covered:
- DataDictionary
  - DataField (continuous, categorical, ordinal)
    - Value
    - Interval
- TransformationDictionary / LocalTransformations
  - DerivedField
- TreeModel
  - SimplePredicate
  - SimpleSetPredicate
- Segmentation *('majorityVote' for Random Forests, 'modelChain' and 'sum' for Gradient Boosting)*
- Regression
  - RegressionTable
    - NumericPredictor
    - CategoricalPredictor
- GeneralRegressionModel *(only linear models)*
  - PPMatrix
    - PPCell
  - ParamMatrix
    - PCell
- NaiveBayesModel
  - BayesInputs
    - BayesInput
      - TargetValueStats
        - TargetValueStat
          - GaussianDistribution
      - PairCounts
        - TargetValueCounts
          - TargetValueCount
  
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

More examples can be found in the subsequent packages: [tree](sklearn_pmml_model/tree), [ensemble](sklearn_pmml_model/ensemble), [linear_model](sklearn_pmml_model/linear_model) and [naive_bayes](sklearn_pmml_model/naive_bayes).

## Development

### Prerequisites

Tests can be run using Py.test. Grab a local copy of the source:

```
$ git clone http://github.com/iamDecode/sklearn-pmml-model
$ cd sklearn-pmml-model
```

create a virtual environment and activating it:
```
$ python3 -m venv venv
$ source venv/bin/activate
```

and install the dependencies:

```
$ pip install -r requirements.txt
```

The final step is to build the Cython extensions:

```
$ python setup.py build_ext --inplace
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

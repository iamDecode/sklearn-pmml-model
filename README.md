<img src="https://user-images.githubusercontent.com/1223300/41346080-c2c910a0-6f05-11e8-89e9-71a72bb9543f.png" width="300">

# sklearn-pmml-model

[![PyPI version](https://badge.fury.io/py/sklearn-pmml-model.svg)](https://badge.fury.io/py/sklearn-pmml-model)
[![codecov](https://codecov.io/gh/iamDecode/sklearn-pmml-model/branch/master/graph/badge.svg?token=CGbbgziGwn)](https://codecov.io/gh/iamDecode/sklearn-pmml-model)
[![CircleCI](https://circleci.com/gh/iamDecode/sklearn-pmml-model.svg?style=shield)](https://circleci.com/gh/iamDecode/sklearn-pmml-model)
[![ReadTheDocs](https://readthedocs.org/projects/sklearn-pmml-model/badge/?version=latest&style=flat)](https://sklearn-pmml-model.readthedocs.io/en/latest/)

A library to effortlessly import models trained on different platforms and with programming languages into scikit-learn in Python. First export your model to [PMML](http://dmg.org/pmml/v4-3/GeneralStructure.html) (widely supported). Next, load the exported PMML file with this library, and use the class as any other scikit-learn estimator.


## Installation

The easiest way is to use pip:

```
$ pip install sklearn-pmml-model
```

## Status
The library currently supports the following models:

| Model                                                  | Classification | Regression | Categorical features |
|--------------------------------------------------------|----------------|------------|----------------------|
| [Decision Trees](sklearn_pmml_model/tree)              | ✅             | ✅         | ✅<sup>1</sup>        |
| [Random Forests](sklearn_pmml_model/ensemble)          | ✅             | ✅         | ✅<sup>1</sup>        |
| [Gradient Boosting](sklearn_pmml_model/ensemble)       | ✅             | ✅         | ✅<sup>1</sup>        |
| [Linear Regression](sklearn_pmml_model/linear_model)   | ✅             | ✅         | ✅<sup>3</sup>        |
| [Ridge](sklearn_pmml_model/linear_model)               | ✅<sup>2</sup> | ✅         | ✅<sup>3</sup>        |
| [Lasso](sklearn_pmml_model/linear_model)               | ✅<sup>2</sup> | ✅         | ✅<sup>3</sup>        |
| [ElasticNet](sklearn_pmml_model/linear_model)          | ✅<sup>2</sup> | ✅         | ✅<sup>3</sup>        |
| [Gaussian Naive Bayes](sklearn_pmml_model/naive_bayes) | ✅             |            | ✅<sup>3</sup>        |
| [Support Vector Machines](sklearn_pmml_model/svm)      | ✅             | ✅         | ✅<sup>3</sup>        |
| [Nearest Neighbors](sklearn_pmml_model/neighbors)      | ✅             | ✅         |                      |
| [Neural Networks](sklearn_pmml_model/neural_network)   | ✅             | ✅         |                      |

<sub><sup>1</sup> Categorical feature support using slightly modified internals, based on [scikit-learn#12866](https://github.com/scikit-learn/scikit-learn/pull/12866).</sub>

<sub><sup>2</sup> These models differ only in training characteristics, the resulting model is of the same form. Classification is supported using `PMMLLogisticRegression` for regression models and `PMMLRidgeClassifier` for general regression models.</sub>

<sub><sup>3</sup> By one-hot encoding categorical features automatically.</sub>
  
## Example
A minimal working example (using [this PMML file](https://github.com/iamDecode/sklearn-pmml-model/blob/master/models/randomForest.pmml)) is shown below:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.ensemble import PMMLForestClassifier
from sklearn_pmml_model.auto_detect import auto_detect_estimator

# Prepare the data
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

# Specify the model type for the least overhead...
#clf = PMMLForestClassifier(pmml="models/randomForest.pmml")

# ...or simply let the library auto-detect the model type
clf = auto_detect_estimator(pmml="models/randomForest.pmml")

# Use the model as any other scikit-learn model
clf.predict(Xte)
clf.score(Xte, yte)
```

More examples can be found in the subsequent packages: [tree](sklearn_pmml_model/tree), [ensemble](sklearn_pmml_model/ensemble), [linear_model](sklearn_pmml_model/linear_model), [naive_bayes](sklearn_pmml_model/naive_bayes), [svm](sklearn_pmml_model/svm), [neighbors](sklearn_pmml_model/neighbors) and [neural_network](sklearn_pmml_model/neural_network).

## Benchmark

Depending on the data set and model, `sklearn-pmml-model` is between 5 and a 1000 times faster than competing libraries, by leveraging the optimization and industry-tested robustness of `sklearn`. Source code for this benchmark can be found in the corresponding [jupyter notebook](benchmark.ipynb). 


### Running times (load + predict, in seconds)
|               |                     | Linear model | Naive Bayes | Decision tree | Random Forest | Gradient boosting |
|---------------|---------------------|--------------|-------------|---------------|---------------|-------------------|
| Wine          | `PyPMML`            | 0.773291     | 0.77384     | 0.777425      | 0.895204      | 0.902355          |
|               | `sklearn-pmml-model`| 0.005813     | 0.006357    | 0.002693      | 0.108882      | 0.121823          |
| Breast cancer | `PyPMML`            | 3.849855     | 3.878448    | 3.83623       | 4.16358       | 4.13766           |
|               | `sklearn-pmml-model`| 0.015723     | 0.011278    | 0.002807      | 0.146234      | 0.044016          |

### Improvement

|               |                    | Linear model | Naive Bayes | Decision tree | Random Forest | Gradient boosting |
|---------------|--------------------|--------------|-------------|---------------|---------------|-------------------|
| Wine          | Improvement        | 133×         | 122×        | 289×          | 8×            | 7×                |
| Breast cancer | Improvement        | 245×         | 344×        | **1,367×**    | 28×           | 94×               |

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

Feel free to make a contribution. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

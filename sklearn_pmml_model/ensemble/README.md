# sklearn-pmml-model.ensemble

This package contains `PMMLForestClassifier` and `PMMLGradientBoostingClassifier`.

## Example
A minimal working example is shown below:

### Random Forest
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


### Gradient boosting

To export using `r2pmml`:

```R
library("xgboost")
library("r2pmml")

data(iris)

iris_X = iris[, 1:4]
iris_y = as.integer(iris[, 5]) - 1
iris.matrix = model.matrix(~ . - 1, data = iris_X)
iris.DMatrix = xgb.DMatrix(iris.matrix, label = iris_y)
iris.fmap = as.fmap(iris.matrix)

# Train a model
iris.xgb = xgboost(data = iris.DMatrix, missing = NULL, objective = "multi:softmax", num_class = 3, nrounds = 13)

# Export the model to PMML
r2pmml(iris.xgb, "iris_xgb.pmml", fmap = iris.fmap, response_name = "Species", response_levels = c("setosa", "versicolor", "virginica"), missing = NULL, compact = TRUE)
```

And import:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.ensemble import PMMLGradientBoostingClassifier

# Prepare data
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

clf = PMMLGradientBoostingClassifier(pmml="models/gb-xgboost-iris.pmml")
clf.predict(Xte)
clf.score(Xte, yte)
```
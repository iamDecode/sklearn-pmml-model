# sklearn-pmml-model.ensemble

This package contains `PMMLKNeighborsClassifier` and `PMMLKNeighborsRegressor`.

## Example
A minimal working example is shown below:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier

# Prepare data
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

clf = PMMLKNeighborsClassifier(pmml="models/knn.pmml")
clf.predict(Xte)
clf.score(Xte, yte)
```
# sklearn-pmml-model.neural_network

This package contains `PMMLMLPClassifier` and `PMMLMLPRegressor`.

## Example
A minimal working example is shown below:

```python
import numpy as np
import pandas as pd
from sklearn_pmml_model.neural_network import PMMLMLPClassifier
from sklearn.datasets import load_iris

# Prepare data
data = load_iris(as_frame=True)
X = data.data
y = pd.Series(np.array(data.target_names)[data.target])
y.name = "Class"

clf = PMMLMLPClassifier(pmml="models/nn-iris.pmml")
clf.predict(X)
```
# sklearn-pmml-model.neighbors

This package contains `PMMLKNeighborsClassifier` and `PMMLKNeighborsRegressor`.

## Example
A minimal working example is shown below:

```python
import pandas as pd
import numpy as np
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier

# Prepare data
df = pd.read_csv('models/categorical-test.csv')
cats = np.unique(df['age'])
df['age'] = pd.Categorical(df['age'], categories=cats).codes + 1
Xte = df.iloc[:, 1:]

clf = PMMLKNeighborsClassifier(pmml="models/knn-clf-pima.pmml")
clf.predict(Xte)
```
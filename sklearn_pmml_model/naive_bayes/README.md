# sklearn-pmml-model.naive_bayes

This package contains the `PMMLGaussianNB` classifier.

## Example
A minimal working example is shown below:

```python
import pandas as pd
from sklearn_pmml_model.naive_bayes import PMMLGaussianNB

# Prepare data
df = pd.read_csv('models/categorical-test.csv')
Xte = df.iloc[:, 1:]

clf = PMMLGaussianNB(pmml="models/nb-cat-pima.pmml")
clf.predict(Xte)
```
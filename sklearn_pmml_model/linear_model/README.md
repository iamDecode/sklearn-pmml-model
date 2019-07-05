# sklearn-pmml-model.linear_model

This package contains the `PMMLLinearRegression` (`lm` in R) as well as `PMMLRidge`, `PMMLLasso` and `PMMLElasticNet` (`glm` and `glmnet` in R).

## Example
A minimal working example is shown below:

```python
import pandas as pd
from sklearn_pmml_model.linear_model import PMMLLinearRegression

# Prepare data
df = pd.read_csv('models/categorical-test.csv')
Xte = df.iloc[:, 1:]

clf = PMMLLinearRegression(pmml="models/linear-regression.pmml")
clf.predict(Xte)
```
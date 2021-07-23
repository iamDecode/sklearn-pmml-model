# sklearn-pmml-model.svm

This package contains the `PMMLLinearSVC`, `PMMLNuSVC` and `PMMLSVC` classifier models, as well as the `PMMLLinearSVR`, `PMMLNuSVR` and `PMMLSVR` regression models. 

## Example
A minimal working example is shown below:

```python
import pandas as pd
from sklearn_pmml_model.svm import PMMLSVC

# Prepare data
df = pd.read_csv('models/categorical-test.csv')
Xte = df.iloc[:, 1:]
Xte = pd.get_dummies(Xte, prefix_sep='')  # create categorical variable

clf = PMMLSVC(pmml="models/svc_cat_pima.pmml")
clf.predict(Xte)
```
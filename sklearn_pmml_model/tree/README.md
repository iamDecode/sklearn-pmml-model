# sklearn-pmml-model.tree

This package contains the `PMMLTreeClassifier`.

## Example
A minimal working example is shown below:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.tree import PMMLTreeClassifier

# Prepare data
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

clf = PMMLTreeClassifier(pmml="models/decisionTree.pmml")
clf.predict(Xte)
clf.score(Xte, yte)
```

To interpret the resulting tree, including categorical spits, we adapted the example from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html:

```python
node_indicator = clf.decision_path(X)
leaf_id = clf.apply(X)

sample_id = 0
# obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample {id}:\n'.format(id=sample_id))
for node_id in node_index:
  # continue to the next node if it is a leaf node
  if leaf_id[sample_id] == node_id:
    continue

  # check if value of the split feature for sample 0 is below threshold
  if isinstance(clf.tree_.threshold[node_id], list):
    threshold_sign = "in"
  elif (X.iloc[sample_id, clf.tree_.feature[node_id]] <= clf.tree_.threshold[node_id]):
    threshold_sign = "<="
  else:
    threshold_sign = ">"

  print("decision node {node} : (X[{sample}, {feature}] = {value}) "
        "{inequality} {threshold})".format(
    node=node_id,
    sample=sample_id,
    feature=clf.tree_.feature[node_id],
    value=X.iloc[sample_id, clf.tree_.feature[node_id]],
    inequality=threshold_sign,
    threshold=str(clf.tree_.threshold[node_id])))
```
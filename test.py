from pypmml import Model
import numpy as np
import pandas as pd
from os import path

# df = pd.read_csv(path.join('models/categorical-test.csv'))
# cats = np.unique(df['age'])
# df['age'] = pd.Categorical(df['age'], categories=cats).codes + 1
# Xte = df.iloc[:, 1:]
# yte = df.iloc[:, 0]
#
# #model = Model.load('tests/neignbors/knn-sklearn2pmml.pmml')
# model = Model.load('models/knn-reg-pima.pmml')
# results = model.predict(Xte)
# print(results)


from sklearn.datasets import load_iris
pd.set_option("display.precision", 16)

data = load_iris(as_frame=True)

X = data.data
y = data.target
y.name = "Class"

#model = Model.load('tests/neignbors/knn-sklearn2pmml.pmml')
model = Model.load('models/nn-iris.pmml')
results = model.predict(X)
print(results.to_string())
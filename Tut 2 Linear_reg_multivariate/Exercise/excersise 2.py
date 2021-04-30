import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from matplotlib import pyplot as plt

data = pd.read_csv('hiring.csv')
print(data)
data['experience'] = data['experience'] .fillna(0)

medvalForTestScore = data['test_score(out of 10)'].mean()
print(medvalForTestScore)
data['test_score(out of 10)'] = data['test_score(out of 10)'].fillna(medvalForTestScore)

print(data)

# lets make a model
mymodel = linear_model.LinearRegression()
mymodel.fit(data.drop('salary($)',axis='columns'),data['salary($)'])
print("model is ready ",mymodel)
print("prediction for 2yr exp , 9 test score , 6 interview score is" , mymodel.predict([[2,9,6]]))
print("prediction for 12yr exp , 10 test score , 10 interview score is" , mymodel.predict([[12,10,10]]))
print(mymodel.predict([[2,10,10]]))
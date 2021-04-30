import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from matplotlib import pyplot as plt

data = pd.read_csv("homeprices.csv")
print(data)
# here  there is one empty cell so we have to handle that

medianval = data['bedrooms'].median()
print(data,medianval)

data['bedrooms'] = data['bedrooms'].fillna(medianval)
print(data)


# lets make a mode
mymodel = linear_model.LinearRegression()
# fit functions takes 2 args 1st is all independent var and 2nd dep var
mymodel.fit(data.drop('price',axis='columns'),data['price'])
# for  x args we gave the whole data set after droping the price col
print("model is tranied",mymodel)

# lets predict
# print(mymodel.predict(3000,3,40)) #wrong
print(mymodel.predict([[3000,3,40]])) #right

print("coff is",mymodel.coef_,"intercept is ",mymodel.intercept_)

# i dont'how to plot the graph

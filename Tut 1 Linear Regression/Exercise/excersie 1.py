import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("canada_per_capita_income.csv")
dataset = dataset
print(dataset,type(dataset))

dataTrainX = dataset[['year']]
datatrainY = dataset['income']
print(dataTrainX,datatrainY)

# lets plot a scatter plot
plt.scatter(dataTrainX,datatrainY,color="red",marker='+')
plt.xlabel("year")
plt.ylabel("income")
# plt.show()


# now make a model
mymodel = linear_model.LinearRegression()
mymodel.fit(dataTrainX,datatrainY)
predictedvalues = mymodel.predict(dataTrainX)
print("mean sq error is ",mean_squared_error(dataTrainX,predictedvalues))
print("predict values",predictedvalues)

# now make a line which best fits

plt.scatter(dataTrainX,datatrainY,color="red",marker='+')
plt.plot(dataTrainX,predictedvalues)
plt.xlabel("year")
plt.ylabel("income")
plt.show()

# lets predict incomee for 2019
print(mymodel.predict([[2016]]))
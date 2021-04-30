import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv("homeprices.csv")


model = linear_model.LinearRegression()

model.fit(df[['area']],df.price)
print(model.predict([[2600]]))
print(model.score(df[['area']],df['price']))


import pickle
#for saving
with open("model_pickle","wb") as file:
    pickle.dump(model,file)

# For loading
with open("model_pickle","rb") as file:
    model_new = pickle.load(file)

print(model_new.predict([[2600]]))

# for large numpy array we shoule use joblib
import joblib

#for saving
joblib.dump(model,"model_joblib")
#for loading
model_new2 = joblib.load('model_joblib')
print(model_new2.predict([[2600]]))
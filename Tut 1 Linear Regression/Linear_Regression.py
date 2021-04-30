import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#loading data frame
df = pd.read_csv("homeprices.csv")
print(df)

#ploting a graph
plt.scatter(df.area,df.price,color="red",marker='+')
plt.xlabel("area sq feet")
plt.ylabel("prices US $")
# plt.show()

reg = linear_model.LinearRegression()

#tranning the model with data
reg.fit(df[['area']],df.price)
# Q why we give df[[area]] not df.area
# ans : because it needs a 2-d array as input

print(reg,reg.coef_,reg.intercept_)
m = reg.coef_
c = reg.intercept_
print("predect value of 3300 is ",reg.predict([[3300]]))


#task
d = pd.read_csv("areas.csv")
# d=d.head(5)
print(d)

predicted_prices = reg.predict(d)
print(predicted_prices)
# adding new col
d['prices'] = predicted_prices
d.to_csv("prediction aj.csv",index=False)

plt.scatter(df.area,df.price,color="red",marker='+')
plt.xlabel("area sq feet")
plt.ylabel("prices US $")
plt.plot(df.area,reg.predict(df[['area']]))
plt.show()

import pandas as pd

df = pd.read_csv("carprices.csv")

print(df.head())
import matplotlib.pyplot as plt

plt.scatter(df['Mileage'],df['Sell Price($)'])
# plt.show()

plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
# plt.show()

X = df[['Mileage','Age(yrs)']]
Y = df['Sell Price($)']
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test = train_test_split(X,Y,test_size = 0.3)

print(len(X_train),len(X_test),len(Y_train),len(Y_test))

from sklearn.linear_model import  LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)
print(model.predict(X_test))
print(model.score(X_test,Y_test))

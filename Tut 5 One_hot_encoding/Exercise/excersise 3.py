import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../ML/5_one_hot_encoding/Exercise/carprices.csv')
dummy = pd.get_dummies(df['Car Model'])
df_new = pd.concat([df,dummy],axis="columns")
df_new = df_new.drop(['Car Model','Mercedez Benz C class'],axis="columns")
X = df_new.drop(['Sell Price($)'],axis="columns")
Y = df['Sell Price($)']
model = LinearRegression()
model.fit(X,Y)
print(model.predict([[59000,5,1,0]]))
print(X.values)
Ynew = model.predict(X.values)
print(Ynew)
print(model.coef_,model.intercept_)
import matplotlib.pyplot as plt
plt.plot(X,Y)
plt.show()

# values to shi aarhi hai pr plot ni ho pa ra hai shi se
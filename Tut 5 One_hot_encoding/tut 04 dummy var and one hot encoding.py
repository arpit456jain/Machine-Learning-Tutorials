# now we predict price of house in two diff areas
# Q : if 3400 sq ft area in west windsor
# Q : if 2800 sq ft area in robbinsville

# Ml model always work in numbers it doesn't know text

# but its not cool to categerise them in number that's why we use One Hot Encoding
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")
print(df)


dummy = pd.get_dummies(df.town)
print(dummy)

# join our dataframe and these dummy var
df_new = pd.concat([df,dummy],axis="columns")
print(df_new)

# now we dont need town col and we have to drop 1 dummy col because of dummy trap
# we will drop town and west windsor

df_new = df_new.drop(['town','west windsor'],axis="columns")
print(df_new)

X = df_new.drop(['price'],axis="columns")
Y = df_new.price
print(X,Y)

model = LinearRegression()
model.fit(X,Y)
print(model.predict([[3400,0,0]]))
print(model.predict([[2800,0,1]]))



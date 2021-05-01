import matplotlib.pyplot as plt
from  sklearn.datasets import load_digits

digits = load_digits()
print(digits) # its  a dictionary
print(dir(digits),digits.images,digits.images[0])

plt.matshow(digits.images[0])
# plt.show()

# for i in range(5):
#     plt.matshow(digits.images[i])

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import train_test_split
X = digits.data
Y = digits.target
X_train , X_test, Y_train , Y_test = train_test_split(X,Y,test_size = 0.2)
print(len(X_train),len(X_test),len(Y_train),len(Y_test))
model.fit(X_train,Y_train)
model.predict(X_test)
print(model.score(X_test,Y_test))
Y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_predicted)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm , annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
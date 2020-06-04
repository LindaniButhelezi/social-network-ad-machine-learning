'''
Analysis of data for a social media advert
using logistic regression
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values #X=[age, salaries]
y = dataset.iloc[:, 4].values #purchase/ no purchase 
X[:,1]=X[:,1]/1000

'''I am dividing salaries by 1000 to make visualisation easier
An alternative to this is normalising the data. but I dont like because it
make the plots difficult to read as ages and salaries will be between 0 and 1
which could confuse some. To learn more read up about standarScalar'''


# Splitthe dataset into the Training set and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap=ListedColormap(colors[:len(np.unique(y_test))])

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.1),
                     np.arange(start = X_set[:, 1].min() -50, stop = X_set[:, 1].max() + 100, step = 0.1))

Z=classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z=Z.reshape(X1.shape)
plt.contourf(X1,X2,Z, alpha=0.4,cmap=cmap)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

def didBuy(i):
    if i==1:
        #y= 1 or y=0 , 1=purchase, 0=no purchase
        return "Purchase Made" 
    else:
        return "No Purchase Made"

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = colors[i], label=didBuy(j))
plt.title('Results of the Logistic Regression Using Training Data')
plt.xlabel('Age')
plt.ylabel("Estimated Salary ( '000 USD)")
plt.legend()

plt.show()


# Visualising test results
X_set, y_set = X_test, y_test
colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap=ListedColormap(colors[:len(np.unique(y_test))])

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.1),
                     np.arange(start = X_set[:, 1].min() -50, stop = X_set[:, 1].max() + 100, step = 0.1))

Z=classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z=Z.reshape(X1.shape)
plt.contourf(X1,X2,Z, alpha=0.4,cmap=cmap)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

def didBuy(i):
    if i==1:
        return "Purchase Made"
    else:
        return "No Purchase Made"

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = colors[i], label=didBuy(j))
plt.title('Results of the Logistic Regression Using Test Data')
plt.xlabel('Age')
plt.ylabel("Estimated Salary ( '000 USD)")
plt.legend()

plt.show()


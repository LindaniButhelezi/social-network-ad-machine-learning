'''
Analysis of data for a social media advert
using logistic regression

This model uses gender, age and salary as X values
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1,2,3]].values #salaries and age are the X values
y = dataset.iloc[:, 4].values #purchase is y value
X[:,2]=X[:,2]/1000

#Transform Male/ Female data to numerical values, male=1, female=0
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_test2[:100]=0
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train[:,[1,2]], y_train
colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmp=ListedColormap(colors[:len(np.unique(y_test))])
sns.heatmap(cm, annot=True,linewidths=0.5,square=True, cmap='plasma')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as pd
from sklearn import datasets

f= pd.read_csv(r'C:\Users\mahmo\Desktop\online_shoppers_intention.csv')





sizes = f['Browser'].value_counts(sort=1)
print(sizes)

'Counts the values and storts them'


y= f['Browser'].values


f.head(5)
f.describe()

sns.scatterplot(x='PageValues',y='BounceRates', data=f, hue='Revenue', palette='prism')
plt.show()

f.drop(['Administrative'],axis=1, inplace = True)


#replaces true false values with 1 and 2
f.Weekend = f.Weekend.replace({True: 1, False: 0})
f.Revenue = f.Revenue.replace({True: 1, False: 0})
print(f)


X, Y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)


train_samples = 100

X_train = X[:train_samples]
X_test = X[train_samples:]
Y_train = Y[:train_samples]
Y_test = Y[train_samples:]







Y=f['Browser'].values
Y=Y.astype('int')




X=f.drop(labels=['Browser'])

from sklearn.model_selection import train_test_split 
X_train , X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.4, random_state = 20)


from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=10, random_state=30)

model.fit(X_train,Y_train)

line1, =plt.plot(X_train,Y_train)
line2, =plt.plot(X_train,Y_train)


plt.show()

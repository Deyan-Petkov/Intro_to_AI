import pandas as pd
import os

path = (r"C:\Users\mahmo\Desktop")


ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)import pandas as pd
import os

path = (r"C:\Users\mahmo\Desktop")


ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)

# Columns and their respective data types
print(df.dtypes)  

#replace True/Flase values with 1 and 0
df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})


#show statistical summary
from pandas import set_option
set_option('display.width',100)
set_option('precision',2)
df.describe()


sizes = df['Browser'].value_counts(sort=1)
print(sizes)

'Counts the values and storts them'


y= df['Browser'].values

#Remove the null values and update the dataset according to the changes
df.dropna(inplace=True)
df.isnull().sum()
print("shape after dropping the missing values\n", df.shape)

#df.info()

#convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
leMonths = LabelEncoder()

#convert/fit the data
"""It is important fit_transform to be run only once, otherwise 
the original values will be lost"""
df['Month'] = leMonths.fit_transform(df['Month'])

#map and show the new and old values (only for better visual understanding)
leMonths_mapping = dict(zip(leMonths.classes_ ,
                            leMonths.transform(leMonths.classes_)))

print(leMonths_mapping)

#remove Other from VisitorType column as it is insignificant number of entries(85)
#and we mainly have two types of visitors - new and returning
df = df[df.VisitorType != 'Other']
#create  LabelEncoder() instance for VisitorType
leVisitor = LabelEncoder()
#convert VisitorType from categorical to numerical
df['VisitorType'] = leVisitor.fit_transform(df['VisitorType'])

leVisitor_mapping = dict(zip(leVisitor.classes_ ,
                                 leVisitor.transform(leVisitor.classes_)))

#print(df)


from sklearn.model_selection import train_test_split

#split the target(y) and the rest of the data(X)
result = []
for x in df.columns:
     if x != 'Revenue':
        result.append(x)
        
X = df[result].values
y = df['Revenue'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 


"""Now we will define and fit our model using sckit-learn pipline."""



"Random Forest Classification"
from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier(n_estimators=100,max_depth=10)

rclf.fit(X_train,y_train)

"""Now we are going to test the model"""

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
y_pred = rclf.predict(X_test)
print('The accuracy is ',accuracy_score(y_test,y_pred))
print('The recall score is ',recall_score(y_test,y_pred))



tb=confusion_matrix(y_test,y_pred)

print(tb)
import matplotlib.pyplot as plt

plt.imshow(tb, cmap='binary')



"Random Forest Regression"
from sklearn.ensemble import RandomForestRegressor
#rcmf = RandomForestRegressor(n_estimators=100,max_depth=10)
#rcmf.fit(X_train,y_train)

#x_pred = rclf.predict(X_test)
#print('The accuracy is ',accuracy_score(y_test,x_pred))
#print('The recall score is ',recall_score(y_test,x_pred))


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



def Scoring(y_pred):
        #The proportion of the variance in the dependent variable predictions 
        #made on basis of the independent variable(s)
        #Best possible score is 1.0 
        print('Test Variance score: %.2f' % r2_score(y_test, y_pred))
        #arithmetic average of the absolute errors (e = y_pred - y). (closer to zero is better)
        print('Mean Absolute Error:%.2f'% mean_absolute_error(y_test, y_pred))
        #the average squared difference between the estimated values and the actual value. (closer to zero is better)
        print('Mean Squared Error:%.2f'% mean_squared_error(y_test, y_pred))
        #square root of the average of squared errors. (closer to zero is better)
        print('\n\n')




def fitTrainScore (dtRegressor):
    dtRegressor.fit(X_train, y_train)
    reg_y_pred = dtRegressor.predict(X_test)
    Scoring(reg_y_pred)


dtRegressor =  RandomForestRegressor(random_state=10)
fitTrainScore(dtRegressor)


"""Our model succed to predit on test data with 90% accuracy."""

import matplotlib.pyplot as plt


plt.figure(figsize=(25,20))
plt.subplot(5, 3, 1)
fig = df.boxplot(column='Administrative')
plt.subplot(5, 3, 2)
fig = df.boxplot(column='Administrative_Duration')
plt.subplot(5, 3, 3)
fig = df.boxplot(column='Informational')
plt.subplot(5, 3, 4)
fig = df.boxplot(column='Informational_Duration')
plt.subplot(5, 3, 5)
fig = df.boxplot(column='ProductRelated')
plt.subplot(5, 3, 6)
fig = df.boxplot(column='ProductRelated_Duration')
plt.subplot(5, 3, 7)
fig = df.boxplot(column='BounceRates')
plt.subplot(5, 3, 8)
fig = df.boxplot(column='ExitRates')
plt.subplot(5, 3, 9)
fig = df.boxplot(column='PageValues')
plt.subplot(5, 3, 10)
fig = df.boxplot(column='SpecialDay')
plt.subplot(5, 3, 11)
fig = df.boxplot(column='Month')
plt.subplot(5, 3, 12)
fig = df.boxplot(column='Region')




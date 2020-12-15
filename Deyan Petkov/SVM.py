# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:03:15 2020

@author: deyan
"""

import os
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

path = "../"

ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)

#set option to print the whole dataset 
pd.set_option("display.max_rows", None, "display.max_columns", None)

#dataset description
from pandas import set_option
set_option('display.width',100)
set_option('precision',2)
df.describe()
df.describe(include = 'all')


#add -1 as a missing value type
df = pd.read_csv(ds, na_values=['NaN','-1'])

#How many sessions ended with transaction or without 
df['Revenue'].value_counts()

#replace True/Flase values with 1/0
df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})

#show null values count
print("count missing/NA values")
df.isnull().sum()

#Remove the null values and update the dataset according to the changes
df.dropna(inplace=True)


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




#------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

#split the target(y) and the rest of the data(X)
result = []
for x in df.columns:
    if x != 'Revenue':
        result.append(x)
        
X = df[result].values
y = df['Revenue'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 



#Creates Confusion matrix using seaborn
def conf_matrix (y_test, y_pred, kernel, gamma, C):
    cm = confusion_matrix(y_test, y_pred)
    fn, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=1, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
    if (len(gamma) > 0):
        plt.title('kernel = '+ kernel + ' gamma =  ' + gamma+ ' C = '+ C)
    else :
        plt.title('kernel = '+ kernel + ' C = '+ C)
    plt.xlabel('Predicted Values')
    plt.ylabel('Test Values')
    plt.show()
    
    
#Normalized version of conf_matrix function
def normalized_conf_matrix (y_test, y_pred, kernel, gamma, C):
    fn, ax = plt.subplots(figsize=(5,5))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, linewidth=1, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
    if (len(gamma) > 0):
        plt.title('kernel = '+ kernel + ' gamma =  ' + gamma+ ' C = '+ C + " Normalized")
    else :
        plt.title('kernel = '+ kernel + ' C = '+ C + " Normalized")
    plt.xlabel('Predicted Values')
    plt.ylabel('Test Values')
    plt.show()


#Combine together some informational metrics
def report (y_test, y_pred):
    print(classification_report(y_test,y_pred))
    aps = average_precision_score(y_test, y_pred)
    print('Precision and Recall : {0:0.2f}'.format(aps))
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))





from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

#Creating plots with the mean accuracy of the linear kernel 
#and C ranging from 1 to 10 
results = []
for i in (np.arange(1,10,2)):
  svc = SVC(kernel = 'linear', C = i)
  svc.fit(X_train,y_train) 
  results.append(svc.score(X_test,y_test)) 
  
plt.plot(np.arange(1,10,2),results) 
plt.show()

#Same as above just this time we plot
#the mean accuracy from C = 0.1 ... 1.0
results = []
for i in (np.arange(0.1, 1, 0.2)):
  svc = SVC(kernel = 'linear', C = i)
  svc.fit(X_train,y_train) 
  results.append(svc.score(X_test,y_test)) 
  
plt.plot(np.arange(0.1, 1, 0.2),results) 
plt.show()

#Fitting the linear kernel with different values for C and displaying 
#relevant data about the predictions
# We can see that 1 is the best value for C, without any trade offs regarding
#generalization
c_values = [0.0001, 0.001, 1.0, 10.0, 100.0]
linear_results = []

for x in c_values: 
      svc = SVC(kernel = 'linear', C = x)
      svc.fit(X_train,y_train) 
      linear_results.append(svc.score(X_test,y_test)) 
      y_pred = svc.predict(X_test)
      report(y_test, y_pred)
      conf_matrix (y_test, y_pred, "linear ", "", str(x))
      normalized_conf_matrix(y_test, y_pred, "linear", "", str(x))
      print("Print x: ", x)
#plt.plot(c_values, linear_results) 
#plt.show()       
      
      
#Changing the decision function to 'ovr'
svm = SVC(kernel = 'linear', C = 1, decision_function_shape = 'ovr').fit(X_train, y_train)
y_pred = svc.predict(X_test)
report(y_test, y_pred)
conf_matrix (y_test, y_pred, "linear ovr", "", "1")
normalized_conf_matrix(y_test, y_pred, "linear ovr", "", "1")

     

#Radial basis function kernel fittet with different gamma values
#and displaing relevant data about the predictions
rbf_results = []
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

for x in gamma_values: 
      svc = SVC(kernel = 'rbf',gamma = x, C = 1)
      svc.fit(X_train,y_train) 
      rbf_results.append(svc.score(X_test,y_test)) 
      y_pred = svc.predict(X_test)
      report(y_test, y_pred)
      conf_matrix (y_test, y_pred, "rbf", str(x), "1")
      normalized_conf_matrix(y_test, y_pred, "rbf", str(x), "1")




#Drop columns 'OperatingSystems', 'Browser', 'TrafficType' as
#we they are not tightly related to our target - Revenue
df = df.drop(columns = ['OperatingSystems', 'Browser', 'TrafficType'])
df.columns

result = []
for x in df.columns:
    if x != 'Revenue':
        result.append(x)
        
X = df[result].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 


#After dropping few columns of the dataset we try the refit and predict again
# but the results are quite same as with having 'OperatingSystems', 'Browser', 'TrafficType'
#in the dataset
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
y_pred = svm.predict(X_test)
conf_matrix(y_test, y_pred, "linear ", "", "1")
normalized_conf_matrix(y_test, y_pred, "linear ", "", "1")
report(y_test, y_pred)


#-----------------------------------------------------------------------------

#Shows how the values in the data are spread out and skewness.
#Columns witth binary values are ommited (Weekend, Revenue...)
#higer C doesn't handle outliers well. 

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

# Ploting some of the most variant data points together
plt.figure(figsize=(15,10))
df.boxplot(['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
       'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues'], notch = True);


    
    
'''======================================================'''

"""We continue with Support Verctor Regression
   still having columns  'OperatingSystems', 'Browser' and 'TrafficType'
   dropped from the dataset. """


from sklearn.preprocessing import StandardScaler
from  sklearn.svm import SVR
from sklearn.metrics import mean_squared_error



#Regression without andy standardization


#c_values = [0.01, 0.1, 1.0, 10, 100]
c_values = [0.01, 0.1, 1.0, 10, 100]


def SVRfit_pred(fitX, fitY, testSet, c_values, description):
    msqe = []    
    for x in ['rbf', 'linear', 'poly', 'sigmoid']:
        print(x)
        for y in c_values:
            print(y)
            svr = SVR(kernel= x, C = y, gamma='auto')
            #svr = SVR(kernel= x, C = 0.1 )
            #svr = SVR(kernel= x, C = 100 )
            svr.fit(fitX, fitY)
            y_pred = svr.predict(testSet)
            #calculate the squared mean error of the model prediction
            msqe.append([x, y, np.sqrt(mean_squared_error(y_test, y_pred))])
    
    results = pd.DataFrame(msqe, columns =["Kernel","C_value", "MSQE"])
    print (description + "  : \n", results)
    

        
SVRfit_pred(X_train, y_train, X_test, c_values, "SVR results without any standardization")
    
'''============================================================'''


#Standardizes the data making it easier to compare 
# and reduces the deviation 
stdScaler = StandardScaler().fit(X_train, y_train)
#We need to transform both the training and testing data
#otherways the testing wouldn't be relevant to the trained data
stand_X_train = stdScaler.transform(X_train)
stand_X_test = stdScaler.transform(X_test)


#the standardized data look
std_X_train = pd.DataFrame(stand_X_train)
pd.DataFrame(stand_X_train).head(20)



#show the datatypes of all columns
df.dtypes

SVRfit_pred(stand_X_train, y_train, stand_X_test, c_values, "SVR results X_train and X_test standardized ")

'''======================================================'''

'''' 
print (y_train.reshape(-1, 1))
stdYScaler = StandardScaler().fit (y_train.reshape(-1,1), y_test.reshape(-1, 1))
stand_y_train = stdYScaler.transform(y_train.reshape(-1, 1))
stand_y_test = stdYScaler.transform(y_test.reshape(-1, 1))

SVRfit_pred(stand_X_train, y_train, stand_X_test, c_values, 
'''


standardized_df = df
stdScaler = StandardScaler().fit(standardized_df)
standardized_df = stdScaler.transform(standardized_df)
standardized_df = pd.DataFrame(standardized_df)
standardized_df.head(5)
standardized_df.isnull().sum()

X = standardized_df.drop(columns = [14]).values
y = standardized_df[14].values
X.shape
y.shape

pd.DataFrame(X).head(5)
pd.DataFrame(y).head(5)
standardized_df.head(5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 


SVRfit_pred(X_train, y_train, X_test, c_values, "All data standardized")



'''================================================================'''
#Take only the columns which doesn't hold binary type of data(categorical) 
#and standardize only them as standardizing a categorical features  will 
#assign a distribution which is not desirable.

df.describe()
numDf = df[['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
       'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues' ]].copy()

#Make dataframe with only categorical features
categDf = df[['SpecialDay', 'Month', 'Region', 'VisitorType', 'Weekend', 'Revenue']].copy()

numDf.head(5)
categDf.head(5)
#fit and transform the dataframe with non categorical data
stdScaler = StandardScaler().fit(numDf)
transformed_numDf = stdScaler.transform(numDf)
#create a dataframe with the transformed features and see how it looks
transformed_numDf = pd.DataFrame(transformed_numDf)
transformed_numDf.describe()

#check for null values before and after concatenating the standardized and the non standardized dataframes
transformed_numDf.isnull().sum()
#concatenate both dataframes (standardized and non standardized features )
standardized_df = pd.concat([categDf, transformed_numDf], axis=1).reindex(categDf.index)
#check again for null values (we have 129 rows)
standardized_df.isnull().sum()
#we drop this columns are they are not so many
standardized_df.shape
standardized_df.dropna(inplace=True)
standardized_df.isnull().sum()

#see how the dataframe looks after these manipulations
standardized_df.shape
standardized_df.head(5)
standardized_df.describe()
standardized_df.dtypes


#split the dataframe into X and Y for training and testing
result = []
for x in standardized_df.columns:
    if x != 'Revenue':
        result.append(x)
        

X = standardized_df[result].values
y = standardized_df['Revenue'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 


#train and test using the preset values for C in c_values and all the kernels
SVRfit_pred(stand_X_train, y_train, stand_X_test, c_values, 
            "SVR results only noncategorical data standardized")
        



"""
y not standardized

SVR results C = 0.1: 
     Kernel  MSQE
0      rbf  0.29
1   linear  0.32
2  sigmoid  9.04
3     poly  0.34
========================
SVR results default: 
     Kernel         MSQE
0      rbf     0.347740
1   linear     0.320182
2  sigmoid  9207.581178
3     poly     0.365643
=======================
SVR results C = 100: 
     Kernel  MSQE
0      rbf  0.29
1   linear  0.32
2  sigmoid  9.04
3     poly  0.34
===================

All data standardized: 


SVR results default:
     Kernel   MSQE
0      rbf   0.80
1   linear   0.90
2  sigmoid  91.45
3     poly   0.94

====================
SVR results only noncategorical data standardized : 
     Kernel   MSQE
0      rbf   0.37
1   linear   0.37
2  sigmoid  95.17
3     poly   0.38

====================
SVR results without any standardization  : 
     Kernel   MSQE
0      rbf   0.29
1   linear   0.32
2  sigmoid  90.86
3     poly   0.33

"""

"""
about 3hr
SVR results X_train and X_test standardized  : 
      Kernel  C_value     MSQE
0       rbf     0.01     0.30
1       rbf     0.10     0.29
2       rbf     1.00     0.29
3       rbf    10.00     0.30
4       rbf   100.00     0.35
5    linear     0.01     0.32
6    linear     0.10     0.32
7    linear     1.00     0.32
8    linear    10.00     0.32
9    linear   100.00     0.32
10     poly     0.01     0.33
11     poly     0.10     0.34
12     poly     1.00     0.33
13     poly    10.00     0.35
14     poly   100.00     0.37
15  sigmoid     0.01     0.79
16  sigmoid     0.10     9.04
17  sigmoid     1.00    90.86
18  sigmoid    10.00   908.32
19  sigmoid   100.00  9207.58


All data standardized  : 
      Kernel  C_value     MSQE
0       rbf     0.01     0.92
1       rbf     0.10     0.82
2       rbf     1.00     0.80
3       rbf    10.00     0.80
4       rbf   100.00     0.88
5    linear     0.01     0.90
6    linear     0.10     0.90
7    linear     1.00     0.90
8    linear    10.00     0.90
9    linear   100.00     0.90
10     poly     0.01     0.98
11     poly     0.10     0.95
12     poly     1.00     0.94
13     poly    10.00     0.90
14     poly   100.00     0.94
15  sigmoid     0.01     1.14
16  sigmoid     0.10     6.04
17  sigmoid     1.00    91.45
18  sigmoid    10.00   914.19
19  sigmoid   100.00  8885.94


SVR results only noncategorical data standardized  : 
      Kernel  C_value     MSQE
0       rbf     0.01     0.37
1       rbf     0.10     0.37
2       rbf     1.00     0.37
3       rbf    10.00     0.38
4       rbf   100.00     0.42
5    linear     0.01     0.37
6    linear     0.10     0.37
7    linear     1.00     0.37
8    linear    10.00     0.37
9    linear   100.00     0.37
10     poly     0.01     0.37
11     poly     0.10     0.37
12     poly     1.00     0.38
13     poly    10.00     0.40
14     poly   100.00     0.41
15  sigmoid     0.01     0.88
16  sigmoid     0.10     9.52
17  sigmoid     1.00    95.17
18  sigmoid    10.00   952.13
19  sigmoid   100.00  9521.39


"""

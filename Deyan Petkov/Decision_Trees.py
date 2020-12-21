
import os
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
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

df.shape
#(12330, 18)

#add -1 as a missing value type
df = pd.read_csv(ds, na_values=['NaN','-1'])

#How many sessions ended with transaction or without 
df['Revenue'].value_counts()
#False    10422
#True      1908

#replace True/Flase values with 1/0
df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})

#show null values count
print("count missing/NA values")
df.isnull().sum()

#Remove the null values and update the dataset according to the changes
df.dropna(inplace=True)
df.shape
#(12283, 18)

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


df['VisitorType'].value_counts()
'''
Returning_Visitor    10504
New_Visitor           1694
Other                   85
'''
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

df = df.drop(columns = ['OperatingSystems', 'Browser', 'TrafficType'])
df.columns


#split the target(y) and the rest of the data(X)
result = []
for x in df.columns:
    if x != 'Revenue':
        result.append(x)
        
X = df[result].values
y = df['Revenue'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 



#=============================

from sklearn.metrics import classification_report   
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



#confusion matrix
def conf_matrix(testData, predData, description):
    cm = confusion_matrix(testData, predData)
    print(cm)
    fn, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=1, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
    plt.title('Decision Tree\n' + description)
    plt.xlabel('Predicted Values')
    plt.ylabel('Test Values')
    plt.show()


#normalized 
def normalized_conf_matrix(testData, predData, description):
    cm = confusion_matrix(testData, predData)
    fn, ax = plt.subplots(figsize=(5,5))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    sns.heatmap(cm_normalized, annot=True, linewidth=1, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
    plt.title('Decision Tree - normalized\n' + description)
    plt.xlabel('Predicted Values')
    plt.ylabel('Test Values')
    plt.show()


#sns.pairplot(df)

stdScaller = StandardScaler()
X_train = stdScaller.fit_transform(X_train)
X_test = stdScaller.transform(X_test)
#default setings
DefDTClassifier = DecisionTreeClassifier(random_state=10)

#=====================Make predictions using Default Classifier Settings
DefDTClassifier.fit(X_train, y_train)
y_pred = DefDTClassifier.predict(X_test)

pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

conf_matrix(y_test, y_pred, 'default settings')
normalized_conf_matrix(y_test, y_pred, 'default settings')

'''              0       1  accuracy  macro avg  weighted avg
precision     0.93    0.54      0.86       0.74          0.87
recall        0.91    0.59      0.86       0.75          0.86
f1-score      0.92    0.57      0.86       0.74          0.87
support    2593.00  457.00      0.86    3050.00       3050.00
cm
[[2365  228]
 [ 186  271]]
normalized
[[0.91207096 0.08792904]
 [0.40700219 0.59299781]]
'''

#===================Optimized Classifier

#Set of hyperparameters to feed the Gridsearch function
parameters = {'criterion':("gini", "entropy"),
              "min_samples_split":np.arange(2,20,2), 
              "min_samples_leaf":list(range(1, 50)),
              "max_depth":(list(range(1, 10))) }

#Loops through the set of parameters fitting the estimator with each of them
#in order to find the best fit. The decision of best hyperparameters is made upon best accuracy results
dt_gridSearch = GridSearchCV(DefDTClassifier, parameters, scoring="accuracy", cv=5)
#Fit the training data to the GridSearchCV
dt_gridSearch.fit(X_train, y_train)
topParam = dt_gridSearch.best_params_#save the parameters which gave best accuracy
print(f"Best paramters: {topParam})")
#Best paramters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 38, 'min_samples_split': 2}
    
#Set the classifeir to use our prefered parameters
dtClassifier = DecisionTreeClassifier(**topParam)
dtClassifier.fit(X_train, y_train)#fit the classifier to the training data

#How well the training data fit
x_train_pred = dtClassifier.predict(X_train)


trainPrecisionScore = precision_score(y_train, x_train_pred)
print(trainPrecisionScore)
#0.7892335766423357
trainAccuracyScore = accuracy_score(y_train, x_train_pred)
print(trainAccuracyScore)
#0.9124398775688675

#accuracy, precisison on the TRAINING DATA
pd.DataFrame(classification_report(y_train, x_train_pred, output_dict=True))

conf_matrix(y_train, x_train_pred, 'Performance over the training data\ncriterion: gini, max_depth: 8, \nmin_samples_leaf: 38, min_samples_split: 2')
normalized_conf_matrix(y_train, x_train_pred, 'Performance over the training data\ncriterion: gini, max_depth: 8, \nmin_samples_leaf: 38, min_samples_split: 2')

"""              0        1  accuracy  macro avg  weighted avg
precision     0.93     0.79      0.91       0.86          0.91
recall        0.97     0.60      0.91       0.79          0.91
f1-score      0.95     0.68      0.91       0.82          0.91
support    7713.00  1435.00      0.91    9148.00       9148.00

confusion matrix 
[[7482  231]
 [ 570  865]]

normalized
[[0.97005056 0.02994944]
 [0.39721254 0.60278746]]
"""

#make predictions upon the testing set of data
y_pred = dtClassifier.predict(X_test)

testPrecisionScore = precision_score(y_test, y_pred)
print(testPrecisionScore)
#0.7057220708446866
testAccuracyScore = accuracy_score(y_test, y_pred)
print(testAccuracyScore)
#0.899672131147541

#accuracy, precision on the Testing data
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

conf_matrix(y_test, y_pred, 'Performance over the testing data\ncriterion: gini, max_depth: 8, \nmin_samples_leaf: 38, min_samples_split: 2')
normalized_conf_matrix(y_test, y_pred, 'Performance over the testing data\ncriterion: gini, max_depth: 8, \nmin_samples_leaf: 38, min_samples_split: 2')

"""              0       1  accuracy  macro avg  weighted avg
precision     0.93    0.71       0.9       0.82          0.89
recall        0.96    0.57       0.9       0.76          0.90
f1-score      0.94    0.63       0.9       0.79          0.90
support    2593.00  457.00       0.9    3050.00       3050.00


confusion_matrix
[[2485  108]
 [ 198  259]]

confuison_matrix_normalized
[[0.9583494  0.0416506 ]
 [0.43326039 0.56673961]]
"""


#Picture the tree
import graphviz 
from sklearn import tree
#Make list with the features except the dependent one
features = list(df.drop('Revenue',axis=1))
print (features)


# not optimized tree graph
dot_data = tree.export_graphviz(DefDTClassifier, 
    out_file=None,
    filled=True, 
    rounded=True,  
    special_characters=True,
    feature_names = features) 
graph = graphviz.Source(dot_data)
graph


# Optimized Classifer Graph
dot_data = tree.export_graphviz(dtClassifier, 
    out_file=None,
    filled=True, 
    rounded=True,  
    special_characters=True,
    feature_names = features)
graph = graphviz.Source(dot_data)
graph





#==================REGRESSION==================

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#from sklearn import metrics



def Scoring(y_prd):
        #The proportion of the variance in the dependent variable predictions 
        #made on basis of the independent variable(s)
        #Best possible score is 1.0 
        print('Test Variance score: %.2f' % r2_score(y_test, y_prd))
        #arithmetic average of the absolute errors (e = y_pred - y). (closer to zero is better)
        print('Mean Absolute Error:%.2f'% mean_absolute_error(y_test, y_prd))
        #the average squared difference between the estimated values and the actual value. (closer to zero is better)
        print('Mean Squared Error:%.2f'% mean_squared_error(y_test, y_prd))
        #square root of the average of squared errors. (closer to zero is better)
        print('Root Mean Squared Error:%.2f'% np.sqrt(mean_squared_error(y_test, y_prd)))
        print('\n\n')
        

#fit, train, score
def fitTrainScore (dtRegressor):
    dtRegressor.fit(X_train, y_train)
    reg_y_pred = dtRegressor.predict(X_test)
    Scoring(reg_y_pred)
        
        

#criterion = mse by default
dtRegressor =  DecisionTreeRegressor(random_state=10)
fitTrainScore(dtRegressor)
'''
Test Variance score: -0.10
Mean Absolute Error:0.14
Mean Squared Error:0.14
Root Mean Squared Error:0.37
'''

dtRegressor =  DecisionTreeRegressor(random_state=10, criterion="mae")
fitTrainScore(dtRegressor)
'''
Test Variance score: -0.20
Mean Absolute Error:0.15
Mean Squared Error:0.15
Root Mean Squared Error:0.39
'''

dtRegressor =  DecisionTreeRegressor(random_state=10, criterion="friedman_mse")
fitTrainScore(dtRegressor)
'''
Test Variance score: -0.08
Mean Absolute Error:0.14
Mean Squared Error:0.14
Root Mean Squared Error:0.37
'''


parameters = {"min_samples_split":np.arange(2,20,2), 
              "min_samples_leaf":list(range(1, 50)),
              "max_depth":(list(range(1, 10))) }

#Lets gridSearch hyperparameters for friedman mean square error criterion as it performed slightly better
dt_gridSearch = GridSearchCV(dtRegressor, parameters, cv=5)
dt_gridSearch.fit(X_train, y_train)
topParam = dt_gridSearch.best_params_
print(f"Best paramters: {topParam})")
#Best paramters: {'max_depth': 6, 'min_samples_leaf': 39, 'min_samples_split': 2}
dtRegressor =  DecisionTreeRegressor(**topParam)

fitTrainScore(dtRegressor)
'''
Test Variance score: 0.42
Mean Absolute Error:0.14
Mean Squared Error:0.07
Root Mean Squared Error:0.27
'''


#=========================

#Tree graph with optimized DecisionTreeRegressor hyperparameters 
dot_data = tree.export_graphviz(dtRegressor, 
    out_file=None,
    filled=True, 
    rounded=True,  
    special_characters=True,
    feature_names = features) 
graph = graphviz.Source(dot_data)
graph




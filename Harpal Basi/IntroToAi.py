import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pandas import set_option
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#------------------------------------------

#Here, we are going to add the dataset path and clean the dataset from null values
# and adjust the values aswell.

#------------------------------------------



path = "/Users/harpal_basi/Intro_to_AI"

ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)
#show all columns and the first 6 lines from each
df.head()

#add -1 as a missing value type
df = pd.read_csv(ds, na_values=['NaN','-1'])

#show statistical summary
from pandas import set_option
set_option('display.width',100)
set_option('precision',2)
df.describe()

#check the null values and the different types of data
df.info()

print("\nmissing/NA values count: ")
df.isnull().sum()


#replace True/Flase values with 1/0
df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})

print("shape before dropping the missing values\n", df.shape)
#Remove the null values and update the dataset according to the changes
df.dropna(inplace=True)
df.isnull().sum()
print("shape after dropping the missing values\n", df.shape)

#convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
leMonths = LabelEncoder()

#convert/fit the data
"""It is important fit_transform to be run only once, otherwise 
the original values will be lost"""
df['Month'] = leMonths.fit_transform(df['Month'])
df.Month.unique()
#map and show the new and old values (only for better visual understanding)
leMonths_mapping = dict(zip(leMonths.classes_ ,
                            leMonths.transform(leMonths.classes_)))
print(leMonths_mapping)
#list(leMonths.inverse_transform([3]))

#show all visitor types
df.VisitorType.unique()
#remove Other from VisitorType column as it is insignificant number of entries
#and we mainly have two types of visitors - new and returning
df = df[df.VisitorType != 'Other']
#create  LabelEncoder() instance for VisitorType
leVisitor = LabelEncoder()
#convert VisitorType from categorical to numerical
df['VisitorType'] = leVisitor.fit_transform(df['VisitorType'])
leVisitor_mapping = dict(zip(leVisitor.classes_ ,
                                  leVisitor.transform(leVisitor.classes_)))
print (leVisitor_mapping)

#The dataset state after all the manipulations
df.head(6)

#------------------------------------------

#Here, we are going to visualise the data. Exploratory data Analysis

#------------------------------------------

# A plot to show Revenue
sns.countplot(df['Revenue'])

# Plot to show Regions
sns.countplot(df['Region'])

# Plot showing VisitorType (Other is omitted due to being insignificant)
sns.countplot(df['VisitorType'])

# A plot showing bounce rates and Revenue
sns.countplot(df['BounceRates'], hue ='Revenue')

#A plot showing Visitor types and Revenue 
sns.countplot(x = 'VisitorType', hue = 'Revenue',data=df)


# Scatter plot that shows the correlation between Bounce rates 
# and Product related duration with a hue of Revenue.
sns.set(rc={'figure.figsize':(10,10)})
sns.scatterplot(x='ProductRelated_Duration',y='BounceRates', data=df, hue='Revenue',palette='prism')
plt.show()

# Scatter plot showing Page values with bounce rates and a hue of Revenue
sns.scatterplot(x='PageValues',y='BounceRates', data=df, hue='Revenue', palette='prism')
plt.show()


#Heatmap showing the correlation between different categories 
# (This was taken from a notebook)

df_interval = df[['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'Revenue']]
correlation_matrix = df_interval.corr()
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(correlation_matrix, annot =True, annot_kws = {'size': 15})
plt.xticks(rotation = 30)

#We can see from the heatmap above that  
#Bounce rates and exit rates are highly correlated. So lets show this

sns.jointplot(data = df, x='ExitRates', y='BounceRates')
plt.show()






#------------------------------------------

#Here, we are going to Apply an algorithm to the data to predict the revenue

#------------------------------------------




#First split the data into Train and test.

y = df['Revenue']
X = df.drop(['Revenue'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Apply the KNN algorithm

kn=KNeighborsClassifier(n_neighbors = 13) #, metric = 'minkowski')
kn.fit(X_train,y_train)
predict_y2=kn.predict(X_test)#Output Prediction
predict_y2

#Checking Accuracy score
KNN_accuracy=accuracy_score(y_test,predic_y2)*100
KNN_accuracy

#Accurate values
confusion_matrix(y_test,predic_y2)

#Classification report
print(classification_report(y_test,predic_y2))


#Determining the best K value using a loop

error_rate = []

for i in range(1,25):
    
    knn=KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    predict_y2=knn.predict(X_test)#Output Prediction
    error_rate.append(np.mean(predict_y2 != y_test))

# Plotting the error rate against the k value to find the best k 
plt.figure(figsize=(10,6))
plt.plot(range(1,25), error_rate)
plt.title('Error rate vs k.value')
plt.xlabel('K')
plt.ylabel('Error Rate')




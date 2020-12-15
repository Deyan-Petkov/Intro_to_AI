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

# Columns and their respective data types
print(df.dtypes)  
  
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

#Dropping OS, Traffic type and Browser from the dataset as it is unuseful.
#This would help with the
#df.drop(['OperatingSystems','Browser','TrafficType'], axis = 1)



#The dataset state after all the manipulations
df.head(6)




#------------------------------------------

#Here, we are going to Apply a KNN and Naive Bayes algorithm to the data to predict the revenue

#------------------------------------------


#First split the data into Train and test.

# Comparing revenue with the other categories of data.
y = df['Revenue']
X = df.drop(['Revenue'],axis=1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# Apply the KNN algorithm

kn=KNeighborsClassifier(n_neighbors = 13 , metric = 'euclidean')
kn.fit(X_train,y_train)
predict_y2=kn.predict(X_test) # Prediction
predict_y2

#Checking Accuracy score
KNN_accuracy=accuracy_score(y_test,predict_y2)*100
KNN_accuracy


#Accurate values
confusion_matrix(y_test,predict_y2)

#Classification report
print(classification_report(y_test,predict_y2))


#Determining the best K value using a loop

error_rate = []

# This for loop calculates the best number of neighbors for the KNN algorithm. 
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

# Naive Bayes Algorithm


from sklearn.naive_bayes import GaussianNB
gauss= GaussianNB()
gauss.fit(X_train,y_train) #Fitting to data

predict_y= gauss.predict(X_test) 
predict_y
confusion_matrix(y_test,predict_y)

#Checking Accuracy
GaussNB_accuracy = accuracy_score(y_test,predict_y)*100
GaussNB_accuracy
print(f'\n Accuracy Score with NB is {GaussNB_accuracy}%.')









#Applying Standardisation and/or Mix max scaling through pca

# Using standard scaler to scale the 
from sklearn.preprocessing import StandardScaler
std_sclr = StandardScaler()
x_train1 = std_sclr.fit_transform(X_train)
x_test1 = std_sclr.transform(X_test)


# For knn we will use PCA to reduce the dimensionality reduction to see if we get a
# better result from our algorithm.
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
x_train_P = pca.fit_transform(x_train1)
x_test_P = pca.transform(x_test1)

# Using the KNN algorithm on the scaled and pca data

KNN_PCA=KNeighborsClassifier(n_neighbors = 15 , metric = 'euclidean')
KNN_PCA.fit(x_train_P,y_train)
predict_y1_P=KNN_PCA.predict(x_test_P) # Prediction
predict_y1_P

#Checking Accuracy score
KNN_accuracy=accuracy_score(y_test,predict_y1_P)*100
KNN_accuracy


# Finding the best Neighbor for the algorithm. Cited this code from Amey Band

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

# Looking at the plot, we can see that 15 would be the best neighbour as the error rate
# is stable and starts to flatten. 

# Using Naive Bayes on the scaled data with PCA
from sklearn.naive_bayes import GaussianNB
gauss_PCA= GaussianNB()
gauss_PCA.fit(x_train_P,y_train) #Fitting to data

predict_y2_P= gauss_PCA.predict(x_test_P) 
predict_y2_P
confusion_matrix(y_test,predict_y)

#Checking Accuracy
GaussNB_accuracy = accuracy_score(y_test,predict_y2_P)*100
GaussNB_accuracy
print(f'\n Accuracy Score with NB is {GaussNB_accuracy}%.')


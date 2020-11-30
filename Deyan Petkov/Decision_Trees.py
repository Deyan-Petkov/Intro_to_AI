# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:03:15 2020

@author: deyan
"""

import os
import pandas as pd

path = "../"

ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)
#show all columns and the first 6 lines from each
df.head(6)

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
"""It is important fit_transform to be run only oce, otherwise 
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
#remove Other from VisitorType column as it is insignificant number of entries(85)
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

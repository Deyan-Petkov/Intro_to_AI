#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:15:37 2020

@author: harpal_basi
"""

import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------

#Here, we are going to visualise the data. Exploratory data Analysis

#------------------------------------------

path = "/Users/harpal_basi/Intro_to_AI"

ds = os.path.join(path,"online_shoppers_intention.csv")
df = pd.read_csv(ds)






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

# We can see from the heatmap above that  
# Bounce rates, exit rates, productRelated and ProductRelated_Duration
# are highly correlated. We visualise these columns to see the correlation

sns.jointplot(data = df, x='ExitRates', y='BounceRates')
plt.show()

plt.figure(figsize=(9,9))
distcols1=df[['ProductRelated', 'ProductRelated_Duration','BounceRates','ExitRates','Revenue']]
sns.pairplot(distcols1,hue='Revenue')

# From this we can see that the lower the bounce and exit rates, the more likely 
# Revenue is True


#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
import os


# In[4]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron


# ## Data Analysis

# In[5]:


test_df = pd.read_csv(r"D:\Projects\Titanic Survival Rate Prediction\test.csv")
train_df = pd.read_csv(r"D:\Projects\Titanic Survival Rate Prediction\train.csv")


# In[6]:


train_df.describe()


# In[7]:


train_df.head()


# In[8]:


train_df.info()


# In[9]:


total = train_df.isnull().sum().sort_values(ascending=False)

percent = train_df.isnull().sum()/train_df.isnull().count()*100

percent_null = (round(percent, 1)).sort_values(ascending=False)


# In[10]:


missing = pd.concat([total, percent_null], axis=1, keys=['Total', 'miss%'])
missing.head(7)


# In[11]:


train_df.columns.values


# ## Data Dependencies

# In[12]:


survived = 'survived'
diceased = 'diceased'
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
men = train_df[train_df['Sex']=='male']
women = train_df[train_df['Sex']=='female']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=18, label = diceased, ax = axes[0], kde =False)
ax.set_title('Female')
ax.legend()
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=30, label = survived, ax = axes[1], kde =False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=30, label = diceased, ax = axes[1], kde =False)
ax.set_title('male')
ax.legend()


# In[13]:


sns.barplot(x='Pclass',y='Survived', data= train_df)
g = sns.FacetGrid(train_df, col="Survived", row="Pclass")
g.map(plt.hist, 'Age', alpha=.5, bins=20)


# In[14]:


grid = sns.FacetGrid(train_df, row="Embarked")
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None)
grid.add_legend()


# In[15]:


grid1 = sns.FacetGrid(train_df, row="Survived")
grid1.map(sns.pointplot, 'SibSp', 'Parch', order=None)
grid1.add_legend()


# In[16]:


sns.barplot(x='SibSp',y='Survived', data= train_df)
g1 = sns.FacetGrid(train_df, col="Survived", row="SibSp")
g1.map(plt.hist, 'Sex', alpha=.5, bins=10)


# In[17]:


axes = sns.factorplot('SibSp','Survived', 
                      data=train_df, aspect = 2.5, )


# In[18]:


axes = sns.factorplot('Parch','Survived', 
                      data=train_df, aspect = 2.5, )


# ### creating a feature to decide number of family members on ship

# In[19]:


data = [train_df, test_df]
for dataset in data:
    dataset['family'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['family'] > 0, 'alone'] = 0
    dataset.loc[dataset['family'] == 0, 'alone'] = 1
    dataset['alone'] = dataset['alone'].astype(int)


# In[20]:


train_df['alone'].value_counts()


# In[21]:


axes = sns.factorplot('family','Survived', 
                      data=train_df, aspect = 2.5, )


# ### Preprocessing

# In[22]:


#train_df = train_df.drop('PassengerId', axis = 1)
#train_df = train_df.drop("PassengerId",axis=0, inplace =True)
#train_df = train_df.drop(columns=["PassengerId"], axis = 1)
train_df.drop('PassengerId', axis=1, inplace=True)


# In[23]:


import re


# In[24]:


deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]
#cabin feature
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)


# In[25]:


train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)


# In[26]:


#embarked
train_df['Embarked'].describe()


# In[27]:


data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna("S")


# In[28]:


# for none values of Age
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)


# In[29]:




train_df["Age"].isnull().sum()


# In[30]:


train_df.info()


# ## converting features to numerics

# In[31]:


#sex
genders = {"female": 0, "male": 1}
data = [train_df, test_df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# In[32]:


#dropping ticket
train_df.drop('Ticket', axis=1, inplace=True)
test_df.drop('Ticket', axis=1, inplace=True)


# In[33]:


#fare
data = [train_df, test_df]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[34]:


#embarked
ports = {"C": 0, "Q": 1, "S": 2}
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[35]:


train_df['Name'].describe()


# In[36]:


#name
data = [train_df, test_df]
titles = {"Mrs":1, "Miss":2,"Mr":3, "Master":4,"Other":5}
for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Miss')
    dataset['Title'] = dataset['Title'].replace(['Countess','Capt','Col','Major','Rev','Lady','Sir','Don','Dr','Dona','Jonkheer'],'Other')
    dataset['Title'] = dataset['Title'].map(titles)


# In[37]:


train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)


# In[38]:


train_df.head(15)


# In[39]:


data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 34), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 34) & (dataset['Age'] <= 42), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 42) & (dataset['Age'] <= 100000), 'Age'] = 6


# In[40]:


train_df['Age'].value_counts()


# In[41]:


train_df.head()


# In[42]:


#fare
train_df['Fare'].value_counts()


# In[43]:


data = [train_df, test_df]
for dataset in data:
    dataset.loc[dataset['Fare'] <= 8,'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 14.5),'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.5) & (dataset['Fare'] <= 30),'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100),'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 250),'Fare'] = 4
    dataset.loc[dataset['Fare'] > 250,'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    


# In[44]:


train_df['Fare'].value_counts()


# In[45]:


train_df.head()


# ## Creating new Features

# In[46]:


# Fare per person
for dataset in data:
    dataset['Fare_per_Person'] = dataset['Fare']/(dataset['family']+1)
    dataset['Fare_per_Person'] = dataset['Fare_per_Person'].astype(int)


# In[47]:


#class times age
for dataset in data:
    dataset['Age_Class'] = dataset['Age']* dataset['Pclass']


# In[48]:


train_df.head(15)


# ## Training different models

# In[49]:


X_train = train_df.drop('Survived', axis = 1)


# In[50]:


X_train


# In[51]:


Y_train = train_df["Survived"]


# In[52]:


Y_train


# In[54]:


X_test = test_df.drop("PassengerId", axis = 1).copy()


# In[55]:


X_test


# In[66]:


sto_grad_des = linear_model.SGDClassifier(max_iter=12, tol=None)
sto_grad_des.fit(X_train, Y_train)
Y_pred = sto_grad_des.predict(X_test)

sto_grad_des.score(X_train, Y_train)

acc_sgd = round(sto_grad_des.score(X_train, Y_train) * 100, 2)


# In[67]:


acc_sgd


# In[78]:


perceptron = Perceptron(max_iter=20)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_percep = round(perceptron.score(X_train, Y_train) * 100, 2)


# In[79]:


acc_percep


# In[86]:


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# In[87]:


acc_log


# In[88]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC


# In[89]:


linear_svm = LinearSVC()
linear_svm.fit(X_train, Y_train)
Y_pred = linear_svm.predict(X_test)
acc_linear_svm= round(linear_svm.score(X_train, Y_train) * 100, 2)


# In[98]:


acc_linear_svm


# In[99]:


random_forest = RandomForestClassifier(n_estimators=150)
random_forest.fit(X_train,Y_train)

Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train,Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[100]:


acc_random_forest


# In[101]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[102]:


acc_decision_tree


# ### K-fold Cross Validation

# In[103]:


from sklearn.model_selection import cross_val_score
random_for = RandomForestClassifier(n_estimators=150)
scores = cross_val_score(random_for, X_train, Y_train, cv=20, scoring = "accuracy")


# In[105]:


print("Scores:",scores)
print("Mean",scores.mean())
print("std. dev",scores.std())


# In[111]:


final_accuracy = (100*(scores.mean()))
print("The final accuracy of our Random Forest Prediction Model is: ", "{:.5f}".format(final_accuracy))


# In[ ]:





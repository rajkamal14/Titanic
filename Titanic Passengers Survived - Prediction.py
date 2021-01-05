#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset

#                                                   **Sinking of Titanic**
# ![Image of Titanic](http://www.titanicuniverse.com/wp-content/uploads/2009/12/titanic-disaster-300x244.jpg)
# 
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# Do a complete analysis on what sorts of people were likely to survive. 

# In[1]:


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as npt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings 
warnings.filterwarnings('ignore')


# In[3]:


from pandas.plotting import scatter_matrix
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[5]:


os.chdir("D:\\Imarticus\\Python Cls\\CSV files-data\\data\\Classification problem")


# In[6]:


os.getcwd()


# In[7]:


dataset = pd.read_csv("titanic.csv")


# In[8]:


dataset.head(2)


# In[9]:


dataset.shape


# In[10]:


dataset.Survived.value_counts()


# In[11]:


dataset.Sex.value_counts()


# In[12]:


dataset.columns


# In[13]:


x= pd.DataFrame()
x['sex']=dataset['Sex']
x['age']=dataset['Age']
x['pclass']=dataset['Pclass']
x['sibSp']=dataset['SibSp']
x['parch']= dataset['Parch']
x['embarked']= dataset['Embarked']


# In[14]:


x.head()


# In[15]:


y=dataset['Survived']
y[:5]


# # Missing Values

# In[16]:


x.isnull().sum()


# In[17]:


x.dtypes


# In[18]:


x['embarked'].mode()


# In[19]:


x['embarked']=x['embarked'].fillna(x.embarked.mode()[0])


# In[20]:


x.hist('age')


# In[21]:


x['age']=x['age'].fillna(x.age.median())


# In[22]:


print(x.sex[:5])
x['sex']=pd.get_dummies(x.sex)['female']
print(x.sex[:5])


# In[23]:


x.pclass.value_counts()


# In[24]:


display (x[:5])
x=x.join(pd.get_dummies(x.pclass, prefix='pclass'))
display(x[:5])


# In[25]:


x= x.drop(['pclass_1','pclass'], axis=1)
display(x[:5])


# In[26]:


x= x.join(pd.get_dummies(x.embarked,prefix='embarked'))
display(x[:5])


# In[27]:


x= x.drop(['embarked','embarked_C'], axis=1)
display(x[:5])


# In[28]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
display(x[:5])
x.age=scaler.fit_transform(x[['age']])
display(x[:5])


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.20, random_state=42)


# In[32]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


model=LogisticRegression()
model.fit(x_train,y_train)

print(model.intercept_)
print(model.coef_)
print(x_train.columns)


# # Prediction

# In[36]:


y_pred = model.predict(x_test)


# In[37]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)
cm1


# In[38]:


(90+54)/(90+15+20+54)


# # K- Fold method for LogisticRegression

# In[39]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=model, X=x_train,y=y_train, cv=20)
accuracies


# # XGBoost

# In[40]:


from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(x_train,y_train)


# In[47]:


Prediction = classifier_xgb.predict(x_test)


# In[48]:


cm2 = confusion_matrix(y_test,Prediction)
cm2


# In[49]:


(92+53)/(92+13+21+53)


# # K- Fold method for XGBoost

# In[46]:


accuracies_xgb = cross_val_score(estimator=classifier_xgb, X=x_train,y=y_train, cv=20)
accuracies_xgb


# In[50]:


accuracies_xgb[8]


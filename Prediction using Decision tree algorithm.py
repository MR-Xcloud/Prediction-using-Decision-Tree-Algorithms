#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING THE LIBRARIES;-
import sklearn.datasets as datasets
import pandas as pd


# In[2]:


#IMPORT THE IRIS DATASETS;-
iris=datasets.load_iris()


# In[4]:


X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.head()


# In[6]:


X.tail()


# In[7]:


X.info()


# In[8]:


X.describe()


# In[9]:


X.isnull().sum()


# In[10]:


Y = iris.target
Y


# In[13]:


#SPLIT DATASET INTO TRAIN AND TEST SETS;-
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[16]:


#DEFINING THE DECISION TREE ALGORITHM
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print("Decision tree Classifier Created Successfully")


# In[17]:



y_predict = dtc.predict(X_test)


# In[18]:


#CONSTRUCTING CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)


# In[19]:


from sklearn import tree
import matplotlib.pyplot as plt


# In[20]:


#VISUALIZING THE DECISION TREE
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(dtc, feature_names = fn, class_names = cn, filled = True);


# In[ ]:





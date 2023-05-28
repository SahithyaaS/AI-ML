#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
d=pd.read_csv('student-por.csv',sep=';')
len(d)


# In[48]:


d.head()


# In[49]:


d['pass']=d.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>=35 else 0,axis=1)
d=d.drop(['G1','G2','G3'],axis=1)
d.head()


# In[50]:


d=pd.get_dummies(d,columns=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
d.head()


# In[51]:


d=d.sample(frac=1)
d_train=d[:500]
d_test=d[500:]

d_train_att=d_train.drop(['pass'],axis=1)
d_train_pass=d_train['pass']

d_test_att=d_test.drop(['pass'],axis=1)
d_test_pass=d_test['pass']

d_att=d.drop(['pass'],axis=1)
d_pass=d['pass']

import numpy as np
print("passing: %d out of %d (%.2f%%)" % (np.sum(d_pass),len(d_pass),100*float(np.sum(d_pass))/len(d_pass)))


# In[52]:


from sklearn import tree
t=tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
t=t.fit(d_train_att,d_train_pass)


# In[53]:


# from sklearn import tree
# import graphviz
# dot_data=tree.export_graphviz(t,out_file=None,label="all",impurity=False,proportion=True,
#                               feature_names=list(d_train_att),class_names=["fail","pass"],
#                               filled=True,rounded=True)
# graph=graphviz.Source(dot_data)
# graph


# In[54]:


t.score(d_test_att,d_test_pass)


# In[55]:


t.score(d_train_att,d_train_pass)


# In[56]:


from sklearn.model_selection import cross_val_score
scores= cross_val_score(t,d_att,d_pass,cv=5)
print("Accuracy:%0.2f (+/- %0.2f)"% (scores.mean(),scores.std()*2))


# In[58]:


for max_depth in range(1,20):
    t=tree.DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
    scores= cross_val_score(t,d_att,d_pass,cv=5)
    print("Max depth= %d,Accuracy = %0.2f (+/- %0.2f)"%(max_depth,scores.mean(),scores.std()*2))


# In[ ]:





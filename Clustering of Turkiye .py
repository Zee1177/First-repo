#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Source: https://www.kaggle.com/honeysingh/clustering-using-turkey-student-data
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("turkiye-student-evaluation_generic.csv")
dataset.head()


# In[3]:


dataset.tail(15)


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


#Lets try to cluster all the students based on the Question responses data.
dataset_questions = dataset.iloc[:,5:33]
dataset_questions.head()


# In[7]:


#lets do a PCA for feature dimensional reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dataset_questions)


# In[8]:


dataset_questions_pca


# In[9]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 99)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[10]:


#Since there are 3 elbows in the graph we can set K = 3
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)


# In[11]:


# Visualising the clusters
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# In[12]:


# 3 clusters of students who have given like Negative, Neutral and Positive feedback
# Checking the count of students in each cluster
import collections
collections.Counter(y_kmeans)


# In[ ]:





import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


import random

import numpy as np

#k mean clustering is a technique in which we have some data say 50 random data points...now lets consider these data points are somewhat categorized eg:money,finance,...related so if we have age as y axis and x as the data set then we can find the mean by
#using the method...in this we have the categorize as different distinguishing mean points(estimate or randomly assigned) then we use func to let these data points match with the actual mean and hence find the mean of the day set
def creteClusterDataSet(N,K):#N is the data set and k is the cluster
    random.seed(1)
    pointsPerCluster=float(N/K)
    X=[]
    for i in range(K):
        incomeCentriod=random.uniform(20000,200000)
        ageCentriod=random.uniform(20,70)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentriod,10000),np.random.normal(ageCentriod,2)])
    X=np.array(X)

    return X

data=creteClusterDataSet(100,5)
model=KMeans(n_clusters=5)

#Scalimg data to normalise it
model=model.fit(scale(data))

#now to look at the data set
print(model.labels_)

plt.figure(figsize=(12,6))
plt.scatter(data[:,0],data[:,1],c=model.labels_.astype(float))#data[:, 0] means get first column for all rows  (this represents X values) data[:, 1] means get second column for all rows  (this represents Y values) c=model. labels_.astype(float) c is for colors of the plot and here we are assigning random colors bases on the number of labels in our model


scale=StandardScaler()
plt.show()

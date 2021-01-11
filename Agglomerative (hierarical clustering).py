import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sys


dataset=pd.read_csv('Mall_Customers.csv')

data=dataset.iloc[:,[3,4]].values


#import scipy.cluster.hierarchy as sch
#
#ded=sch.dendrogram(sch.linkage(data,method='ward'))


from sklearn.cluster import AgglomerativeClustering

cl=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

cl.fit(data)
pred=cl.fit_predict(data)

plt.scatter(data[pred==0,0],data[pred==0,1],color='r')    
plt.scatter(data[pred==1,0],data[pred==1,1],color='b')  
plt.scatter(data[pred==2,0],data[pred==2,1],color='g')  
plt.scatter(data[pred==3,0],data[pred==3,1],color='brown')  
plt.scatter(data[pred==4,0],data[pred==4,1],color='black')  













'''
@author: CSEMN(Mahmoud Nasser)
@since : 30 MAR 2022
'''

import pandas as pd
import matplotlib.pyplot as plt

col_names=['timeStamp','Gender','Grade','Age','Length','Weight','ShoesSize']
dataset = pd.read_csv("../human_features.csv",names= col_names,skiprows=(0,))

#Pick Features
x = dataset.iloc[:,[4,5]].values  # length, weight

#Visualize data
plt.scatter(x[:,0],x[:,1])
plt.xlabel("Lenght")
plt.ylabel("Weight")
plt.show()

#Claculate best K using elbow method
from sklearn.cluster import KMeans
k_test_range = range(2,20)
distorsions = []
for k in k_test_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    distorsions.append(kmeans.inertia_)

from kneed import KneeLocator
kneedle = KneeLocator(k_test_range,distorsions, S=1.0, curve="convex", direction="decreasing")
print('Elbow boint : '+str(kneedle.elbow))
kneedle.plot_knee()

#Clustring
kmeans = KMeans(n_clusters=kneedle.elbow)
kmeans.fit(x)

df = pd.DataFrame({'Length': x[:,0], 'Weight': x[:,1],'Cluster': kmeans.labels_})

#Visualize Clusters
plt.scatter(x[:,0],x[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], s=100,color='black',label='Centroids')
plt.xlabel("Lenght")
plt.ylabel("Weight")
plt.legend()
plt.show()





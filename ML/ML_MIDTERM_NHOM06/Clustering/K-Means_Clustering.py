import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import numpy as np
from sklearn import preprocessing

customer_data = pd.read_csv('dataset4clustering_student.csv', delimiter=',')

## Lấy dữ liệu của các cột có số chỉ định
X = customer_data.iloc[:, [2, 6,8]].values
le =preprocessing.LabelEncoder()
le.fit(X[:,2])
X[:,2]=le.transform(X[:,2])

#print(data.shape[0])

#In ra số rows và columns
#print(X.shape)

# Tìm số cụm tối ưu để phân loại k-mean
wcss = []
for i in range(1, 14):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, max_iter=300,n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# thể hiện kết quả tìm dc lên biểu đồ 
plt.plot(range(1, 14), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# tạo một kmeans classifier với số cluster là 6
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42, max_iter=300,n_init = 10)
y_kmeans = kmeans.fit_predict(X)
#print(X)

# hiện thị các cụm
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='orange', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='violet', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='gray', label='Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=100, c='pink', label='Cluster 6')

# Vẽ đồ thị các cụm trên
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=300, c='black', label='Centroids')
# tên trục X trên đồ thị
plt.xlabel('X')
# tên trục Y trên đồ thị
plt.ylabel('Y')

plt.legend()
plt.show()

# phan cum 3d  
kmeans_plot= KMeans(n_clusters=3, init='k-means++', random_state=42, max_iter=300,n_init = 10)
result = kmeans.fit_predict(X)
# center = kmeans_plot.cluster_centers_
fig =plt.figure()
ax=plt.axes(projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],alpha=0.5,marker='o',c=kmeans_plot.labels_,s=15)
plt.title('Clustering')
ax.set_xlabel('X')
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# ax.scatter(center[:,0], center[:,1], center[:,2], s = 300, c = 'r', marker='*', label = 'Centroid')
plt.show()
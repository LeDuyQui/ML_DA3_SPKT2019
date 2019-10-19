import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

import numpy as np
#load data
loaddata_data=pd.read_csv('Dataset\hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv', delimiter=',')
#phân data lấy cột 1 2 tương ứng giá trị lấy vào mảng
data = loaddata_data.iloc[:,[2,3,4]].values

#xử lý dữ liệu ký tự nếu có.

#le =preprocessing.LabelEncoder()
#le.fit(data[:,0])
#data[:,0]=le.transform(data[:,0])
#le.inverse_transform chuyển về dữ liệu cũ nếu cần
#setup 1 khung xy

plt.figure(figsize=(8,5))
plt.xlabel('x')#nhãn x
plt.ylabel('distance')#nhãn y
plt.title('DG')# tên
#vẽ dendrogram
#gọi thư viện shc từ scipy.cluster để vẽ dendrogram lấy từ giá trị trả về từ phương thức linkage căn cứ vào từ dữ liệu dataset và method để  single tim khoản cách nhỏ nhất giữa các điểm dữ liệu
shc.dendrogram(shc.linkage(data,method='ward'))
plt.axhline(2)
plt.show()
#phân cụm thành 2 cụm tính theo khoản cách 2 điểm dữ liệu theo công thức euclid và linkage method là single cách tính toán khoản cách giữa các cụm
cluster =AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward' )
cluster.fit(data)#phân tên của mỗi cụm mà điểm dữ liệu thuộc về
print('Labels:',cluster.labels_)

#vẽ mô hình phân cụm 2d

plt.figure(figsize=(8,5))

plt.xlabel('x')
plt.ylabel('y')
plt.title('DG')
colors= 10*['r','g','b']
for i in range(len(data)): 
    plt.scatter(data[i][0],data[i][1],c=colors[cluster.labels_[i]],marker='o')
plt.show()  

#vẽ mô hình phân cụm 3d.

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.grid(False)
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], data[i][2], c=colors[cluster.labels_[i]], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
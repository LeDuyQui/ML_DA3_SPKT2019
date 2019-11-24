import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

#from sklearn.cluster import MeanShift
import numpy as np

## Đọc file csv
customer_data = pd.read_csv('dataset4clustering_student.csv', delimiter=',')

## Lấy dữ liệu của các cột có số chỉ định
data = customer_data.iloc[:, [2,6,8]].values
le =preprocessing.LabelEncoder()
le.fit(data[:,2])
data[:,2]=le.transform(data[:,2])
##Tạo kích thước cho khung hình
plt.figure(figsize=(8, 5))
##Gán tên cho trục y
plt.ylabel("Y")
##Gán tên cho trục x
plt.xlabel("X")
##Gán tiêu đề cho Dendrograms
plt.title("Dendrograms")

# vẽ cấu trúc cây Dendrogram
shc.dendrogram(shc.linkage(data, method='ward'))
plt.axhline(35)#ngan cum dai nhat
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit(data)
result_hc=cluster.fit_predict(data)
print("Labels:", cluster.labels_)

# vẽ mô hình phân cụm 2D
# plt.figure(figsize=(8, 5))
# plt.ylabel("Y")
# plt.xlabel("X")
# colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
# for i in range(len(data)):
#     plt.scatter(data[i][0], data[i][1], c=colors[cluster.labels_[i]], marker='o')
# plt.show()
#phan cum 3 d
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.grid(False)
colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], data[i][2], c=colors[cluster.labels_[i]], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
#Calculate precision and recal
#from sklearn.metrics import classification_report
#print(classification_report(testing_label, predicting_label))
customer_data['cluster']= result_hc
customer_data.to_csv("dataset_cluster_hc.csv",index= False)



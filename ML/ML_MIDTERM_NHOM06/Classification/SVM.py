import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


# docdulieu 
DV=pd.read_csv('dataset_for_classification_cmc.csv', encoding='utf-8')

# chuyen du lieu chu thanh so
le =preprocessing.LabelEncoder()
(DV['education'])=le.fit_transform(DV['education'])
(DV['spouse_education'])=le.fit_transform(DV['spouse_education'])
(DV['religion'])=le.fit_transform(DV['religion'])
(DV['use_insurrance'])=le.fit_transform(DV['use_insurrance'])
(DV['now_working'])=le.fit_transform(DV['now_working'])
(DV['spouse_occupation'])=le.fit_transform(DV['spouse_occupation'])

X=DV.iloc[:,:-1].values
Y=DV.iloc[:,9].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3) # 70% training and 30% test
# SVM = SVC(kernel='linear')
# SVM.fit(X_train,Y_train)
# predicting_label = SVM.predict(X_test)
# print(confusion_matrix(Y_test,predicting_label))
# print(classification_report(Y_test,predicting_label))
# print("Accuracy: ", metrics.accuracy_score(Y_test,predicting_label))
# validation
SVM = SVC(kernel='linear')
cv_scores = cross_val_score(SVM, X_train, Y_train, cv=5)
print("Ket qua: ", cv_scores)
print("NP: ",np.mean(cv_scores))
# # veplot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.grid(False)
# for i in range(0, len(Y_train)):
#     if Y_train[i] == 0:
#         ax.scatter(X_train[i][2], X_train[i][3], X_train[i][7],
#                     c='black', s=10, cmap='viridis')
#     else:
#         ax.scatter(X_train[i][2], X_train[i][3], X_train[i][7],
#                     c='red', s = 10, cmap = 'viridis')

# for i in range(0, len(X_test)):
#     if predicting_label[i] == 0:
#         ax.scatter(X_test[i][2], X_test[i][3], X_test[i][7],
#                     c='blue', s=10, cmap='viridis')
#     else:
#         ax.scatter(X_test[i][2], X_test[i][3], X_test[i][7],
#                     c='green', s = 10, cmap = 'viridis')

# plt.title('SVM')
# ax.set_xlabel('Spouse Education')
# ax.set_ylabel('Number Of Student')
# ax.set_zlabel('Index Living Stadard')
# plt.show()



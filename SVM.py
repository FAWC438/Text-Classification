"""
SVM建模与评价
"""

import pickle
import time
import redis

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

redis_conn = redis.Redis(host='127.0.0.1', port=6379, db=0)

coo_test = pickle.loads(redis_conn.get('coo_test'))
# print(coo_test)
coo_train = pickle.loads(redis_conn.get('coo_train'))
# print(coo_train)
class_arr = np.array([int(i / 5000) for i in range(50000)])

# params = {"gamma": [0.001, 0.01, 0.1, 1, 10], "C": [0.001, 0.01, 0.1, 1, 10]}
# model = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=params, scoring='accuracy', cv=5, n_jobs=1)  # 网格搜索自动调参
model = SVC(kernel='rbf', C=5, gamma=0.001)
start = time.time()
model.fit(coo_train.tocsr(), class_arr)
# print("Best parameters:{}".format(model.best_params_))    # C:0.01
# print("Best score on train set:{:.2f}".format(model.best_score_))
end = time.time()
print('Train time: %s Seconds' % (end - start))
start = time.time()
pre = model.predict(coo_test.tocsr())
end = time.time()
print('Test time: %s Seconds' % (end - start))
print(pre)
redis_conn.set('SVM_pre_model', pickle.dumps(pre))

# 混淆矩阵
C = metrics.confusion_matrix(class_arr, pre)
confusion_matrix = pd.DataFrame(C, columns=class_list.values(),
                                index=class_list.values())
confusion_matrix.to_csv('Confusion_Matrix_SVM.csv')
redis_conn.set('confusion_matrix_SVM', pickle.dumps(C))
print("混淆矩阵为：\n", C)
# 计算准确率（accuracy）
accuracy = metrics.accuracy_score(class_arr, pre)
print("准确率为：\n", accuracy)
# 计算精确率（precision）
precision = metrics.precision_score(class_arr, pre, average=None)
print("精确率为：\n", precision)
print('均值{:.4f}\n'.format(sum(precision) / 10))
# 计算召回率（recall）
recall = metrics.recall_score(class_arr, pre, average=None)
print("召回率为：\n", recall)
print('均值{:.4f}\n'.format(sum(recall) / 10))
# 计算F1-score（F1-score）
F1_score = metrics.f1_score(class_arr, pre, average=None)
print("F1值为：\n", F1_score)

cp = metrics.classification_report(class_arr, pre, output_dict=True)
print("---------------分类报告---------------\n", cp)
df = pd.DataFrame(cp).transpose()
df.to_csv("SVM_classification_report.csv", index=True)

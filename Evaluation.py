"""
Part-7:贝叶斯模型评价
"""

import pandas as pd

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

df = pd.read_csv('Confusion_Matrix.csv', index_col=0)
print(df)

acc_sum = 0
all_sum = 0

for tup in df.itertuples():
    acc_sum += df.at[tup[0], tup[0]]
    all_sum += df[tup[0]].sum()

recall_all = 0
precision_all = 0
f1_score_all = 0
right_data = 0
class_index = 0
classification_report_arr = [[] for i in range(12)]
for i, j in class_list.items():
    print('-' * 15 + i + '-' * 15)
    right_data += df.at[j, j]
    recall = df.at[j, j] / df.loc[j, :].sum()
    recall_all += recall
    precision = df.at[j, j] / df[j].sum()
    precision_all += precision
    f1_score = 2 * (recall * precision) / (recall + precision)
    f1_score_all += f1_score
    classification_report_arr[class_index] = [precision, recall, f1_score, 5000]
    print('召回率：' + '{:.4f}'.format(recall))
    print('精确率：' + '{:.4f}'.format(precision))
    print('F1值 ：' + '{:.4f}'.format(f1_score))
    class_index += 1
print('平均召回率：' + '{:.4f}'.format(recall_all / 10))
print('平均精确率：' + '{:.4f}'.format(precision_all / 10))
print('平均F1值：' + '{:.4f}'.format(f1_score_all / 10))
accuracy = right_data / 50000
print('平均准确率：' + '{:.4f}'.format(accuracy))

classification_report_arr[10] = [accuracy for i in range(4)]
classification_report_arr[11] = [precision_all / 10, recall_all / 10, f1_score_all / 10, 5000]
index_col = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'accuracy', 'avg']
index_raw = ['precision', 'recall', 'f1-score', 'support']
df = pd.DataFrame(data=classification_report_arr, columns=index_raw, index=index_col)
df.to_csv("Bays_classification_report.csv", index=True)

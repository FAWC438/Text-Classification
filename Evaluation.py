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
print('总体准确率：' + str(acc_sum / all_sum))

for i, j in class_list.items():
    print('-' * 15 + i + '-' * 15)
    recall = df.at[j, j] / df.loc[j, :].sum()
    precision = df.at[j, j] / df[j].sum()
    print('召回率：' + '{:.4f}'.format(recall))
    print('精确度：' + '{:.4f}'.format(precision))
    print('F1值 ：' + '{:.4f}'.format(2 * (recall * precision) / (recall + precision)))
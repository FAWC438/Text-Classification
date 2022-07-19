import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.style.use('Solarize_Light2')

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

if __name__ == '__main__':
    df_bays = pd.read_csv('Bays_classification_report.csv', index_col=0)
    df_svm = pd.read_csv('SVM_classification_report.csv', index_col=0)
    df_lr = pd.read_csv('LR_classification_report.csv', index_col=0)
    precision_data = [df_bays['precision'].tolist()[:10], df_svm['precision'].tolist()[:10],
                      df_lr['precision'].tolist()[:10]]
    recall_data = [df_bays['recall'].tolist()[:10], df_svm['recall'].tolist()[:10], df_lr['recall'].tolist()[:10]]
    f1_score_data = [df_bays['f1-score'].tolist()[:10], df_svm['f1-score'].tolist()[:10],
                     df_lr['f1-score'].tolist()[:10]]
    draw_data = {'precision': precision_data, 'recall': recall_data, 'f1_score': f1_score_data}

    for tag, data in draw_data.items():
        fig, ax = plt.subplots()

        ax.plot(np.arange(10), data[0], marker='o')
        avg_0 = sum(data[0]) / 10
        # for j in range(6):
        #     ax.text(j, class_data[0][j] - 20, '{:.0f}'.format(class_data[0][j]), size=13)

        ax.plot(np.arange(10), data[1], marker='o')
        avg_1 = sum(data[1]) / 10
        # for j in range(6):
        #     ax.text(j, class_data[1][j] - 20, '{:.0f}'.format(class_data[1][j]), size=13)

        ax.plot(np.arange(10), data[2], marker='o', color='r')
        avg_2 = sum(data[2]) / 10
        # for j in range(6):
        #     ax.text(j, class_data[2][j] - 20, '{:.0f}'.format(class_data[2][j]), size=13)
        # for i in range(2016, 2022):
        #     data_class_0.append(len(df_class_0[str(i)]))
        # print(data_class_0)
        ax.legend(
            ['Bays: {:.2f} (avg)'.format(avg_0), 'SVM: {:.2f} (avg)'.format(avg_1), 'LR: {:.2f} (avg)'.format(avg_2)])
        ax.set_xticks([i for i in range(10)])
        ax.set_xticklabels(class_list.keys())
        ax.set_xlabel('新闻类型')
        ax.set_ylabel(tag)
        ax.set_title('不同类型新闻' + tag + '变化')

        plt.savefig('IMG/不同类型新闻' + tag + '变化.png')
        plt.show()

"""
Part-5:贝叶斯分类器
"""
import pickle
import time
from collections import Counter

import redis
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

redis_conn = redis.Redis(host='127.0.0.1', port=6379, db=0)

word_df = pd.DataFrame()
bayes_df = pd.DataFrame()
df_sum = 1
csc_test = pickle.loads(redis_conn.get('coo_test')).tocsr()
value = csc_test.data
column_index = csc_test.indices
row_pointers = csc_test.indptr
key_words = list(pickle.loads(redis_conn.get('all_key_words_counter_obj')).keys())


def train_bays():
    """
    贝叶斯m-估计矩阵构建

    :return: None
    """
    global word_df, key_words, df_sum, bayes_df
    contents = {}
    word_counters = {}
    for CLASS_NAME_EN in class_list.values():
        contents[CLASS_NAME_EN] = redis_conn.get('data_test_all' + CLASS_NAME_EN).decode()
        word_counters[CLASS_NAME_EN] = Counter(contents[CLASS_NAME_EN].split())
    # 3685
    word_df = pd.DataFrame(np.zeros((6020, 10)), columns=class_list.values(),
                           index=key_words)
    bayes_df = pd.DataFrame(np.zeros((6020, 10)), columns=class_list.values(),
                            index=key_words)

    # 构造关键词词频矩阵
    for CLASS_NAME_EN in class_list.values():
        TF_dic = dict(word_counters[CLASS_NAME_EN])
        for tup in word_df.itertuples():
            if TF_dic.get(tup[0]) is None:  # tup[0]为选定关键词
                continue
            else:
                word_df.at[tup[0], CLASS_NAME_EN] = TF_dic.get(tup[0])
    df_sum = np.array(word_df).sum()
    word_df.to_csv('TF_Matrix.csv')

    # 构造条件概率矩阵
    for tup in bayes_df.itertuples():
        for CLASS_NAME_EN in class_list.values():
            # m-估计
            bayes_df.at[tup[0], CLASS_NAME_EN] = (word_df.at[tup[0], CLASS_NAME_EN] + 1) / (
                    word_df[CLASS_NAME_EN].sum() + df_sum)
    bayes_df.to_csv('Bayes.csv')


def bays(text_pos: int):
    """
    后验概率测试数据集

    :param text_pos:单词位置（偏移量）
    :return:预测类别
    """
    global word_df, bayes_df
    v_NB = {}

    for CLASS_NAME_EN in class_list.values():
        v_NB[CLASS_NAME_EN] = 1
        for v in column_index[row_pointers[text_pos]:row_pointers[text_pos + 1]]:
            # print(np.array(df).sum())
            w = key_words[v]
            v_NB[CLASS_NAME_EN] *= bayes_df.at[w, CLASS_NAME_EN]

    # print(v_NB)
    res = sorted(v_NB.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return res[0][0]


if __name__ == '__main__':
    start = time.time()
    train_bays()
    end = time.time()
    train_time = end - start
    confusion_matrix = pd.DataFrame(np.zeros((10, 10)), columns=class_list.values(),
                                    index=class_list.values())

    start = time.time()
    class_index = 0
    for class_name_en in class_list.values():
        for i in range(5000):
            s = bays(class_index * 5000 + i)
            print('class:' + class_name_en + ' pre:' + s + ' id:' + str(i))
            # 混淆矩阵中，列（class_name_en）为真实值，行（s）为贝叶斯模型的预测值
            confusion_matrix.at[class_name_en, s] += 1
        class_index += 1
    end = time.time()
    print('\n\nTrain time: %s Seconds' % train_time)
    print('Test time: %s Seconds' % (end - start))
    print(confusion_matrix)
    confusion_matrix.to_csv('Confusion_Matrix.csv')

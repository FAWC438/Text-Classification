"""
Part-3:建立词典
"""
import os
import pickle
from collections import Counter

import jieba.analyse
import redis

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

redis_conn = redis.Redis(host='127.0.0.1', port=6379, db=0)
contents = {}
word_counters = {}


def init():
    for CLASS_NAME_EN in class_list.values():
        contents[CLASS_NAME_EN] = redis_conn.get('data_test_all' + CLASS_NAME_EN).decode()
        word_counters[CLASS_NAME_EN] = Counter(
            contents[CLASS_NAME_EN].split())  # 实际上，基于Counter数据结构的word_counter就是词频统计结果


def chi_square(word: str, class_str: str):
    parm_A = 0
    parm_B = 0
    parm_C = 0
    parm_D = 0
    for CLASS_NAME_EN in class_list.values():
        with open('data_train/' + CLASS_NAME_EN + '/all.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # all文件中，一行代表一个文件
            if CLASS_NAME_EN == class_str:
                for j in lines:
                    if word in j.split():
                        parm_A += 1
                    else:
                        parm_C += 1
            else:
                for j in lines:
                    if word in j.split():
                        parm_B += 1
                    else:
                        parm_D += 1
    # print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s' % kf)
    # 返回卡方值，卡方值越大说明相关性越高，该词在该类中越关键
    if parm_B == 0:
        return 1
    return (parm_A + parm_B + parm_C + parm_D) * ((parm_A * parm_D - parm_B * parm_C) ** 2) / (
            (parm_A + parm_C) * (parm_A + parm_B) * (parm_D + parm_B) * (parm_C + parm_D))


if __name__ == '__main__':
    init()  # 将分词后的训练数据从redis数据库读取到内存中

    # 利用jieba内置的TF/IDF方法加权，排序得到其前1000给关键词
    all_key_words_counter = Counter()
    for class_name_en in class_list.values():
        topK = 1000
        if len(word_counters[class_name_en]) < topK:
            topK = len(word_counters[class_name_en])
        tags = jieba.analyse.extract_tags(contents[class_name_en], topK=topK,
                                          withWeight=True)  # TF/IDF加权
        tags_dic = dict(tags)
        index = 0
        for tag, value in tags_dic.items():
            print("class: %s index: %d tag: %s\t\t weight: %f" % (class_name_en, index, tag, value))
            index += 1
            # tags_dic[tag] = value * chi_square(tag, class_name_en)
            tags_dic[tag] = value

        key_words_counter_obj = Counter(tags_dic)  # 以TF/IDF加权的关键词
        # redis中存储每种类型的关键词
        redis_conn.set(class_name_en + '-key_words_counter_obj', pickle.dumps(key_words_counter_obj))
        all_key_words_counter += key_words_counter_obj

    redis_conn.set('all_key_words_counter_obj', pickle.dumps(all_key_words_counter))  # redis中存储全部关键词（共6020个）

"""
Part-2：
从原始数据中提取名称同时删除停用词，再写入文件中
"""

import redis
import jieba
import jieba.posseg as ps

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

with open('stop_words_ch.txt', 'r', encoding='GBK') as file:
    stop_words = [i.strip() for i in file.readlines()]
with open('stop_sign.txt', 'r', encoding='utf-8') as file:
    stop_sign = [i.strip() for i in file.readlines()]

redis_conn = redis.Redis(host='127.0.0.1', port=6379, db=0)


def is_stop_word(word_str: str):
    if word_str in stop_words:
        return True
    for i in word_str:
        if i in stop_sign:
            return True
    return False


def creat_texts():
    for class_name, class_name_en in class_list.items():
        for i in range(5000):
            print(class_name + ':' + str(i))
            # 生成测试集
            string_to_write = ''
            with open('source_data_test/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                lines = f.read()
                words = ps.cut(lines, use_paddle=True)  # jieba分词
                for word, flag in words:
                    # 若为名词且不在停用词表中，则加入待写入串
                    if flag == 'n' and not is_stop_word(word):
                        string_to_write += (word + ' ')

            redis_conn.set('data_test-' + class_name_en + '-' + str(i), string_to_write)  # 存储到redis中
            redis_conn.append('data_test_all' + class_name_en, string_to_write)

            # 生成训练集
            string_to_write = ''
            with open('source_data_train/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                lines = f.read()
                words = ps.cut(lines, use_paddle=True)  # jieba分词
                for word, flag in words:
                    # 若为名词且不在停用词表中，则加入待写入串
                    if flag == 'n' and not is_stop_word(word):
                        string_to_write += (word + ' ')
            redis_conn.set('data_train-' + class_name_en + '-' + str(i), string_to_write)  # 存储到redis中
            redis_conn.append('data_train_all' + class_name_en, string_to_write)


if __name__ == '__main__':
    # jieba.enable_paddle()  # 启动paddle模式
    creat_texts()

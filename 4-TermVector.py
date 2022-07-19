"""
Part-4:生成词向量/词袋模型
"""
import pickle
import redis
import numpy as np
from scipy.sparse import coo_matrix

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

redis_conn = redis.Redis(host='127.0.0.1', port=6379, db=0)

key_words = list(pickle.loads(redis_conn.get('all_key_words_counter_obj')).keys())
# print(len(key_words)) # 6020
# 关键词共6020个

test_arr = np.zeros(shape=(50000, 6020))
train_arr = np.zeros(shape=(50000, 6020))
test_index = 0
train_index = 0

for class_name_en in class_list.values():
    for i in range(5000):
        print(class_name_en + ':' + str(test_index))
        text = redis_conn.get('data_test-' + class_name_en + '-' + str(i)).decode()
        for w in text.split():
            if w not in key_words:
                continue
            else:
                index = key_words.index(w)
                test_arr[test_index][index] += 1
        test_index += 1

    for i in range(5000):
        print(class_name_en + ':' + str(train_index))
        text = redis_conn.get('data_train-' + class_name_en + '-' + str(i)).decode()
        for w in text.split():
            if w not in key_words:
                continue
            else:
                index = key_words.index(w)
                train_arr[train_index][index] += 1
        train_index += 1

# 用稀疏矩阵存储
coo_test = coo_matrix(test_arr)
redis_conn.set('coo_test', pickle.dumps(coo_test))

coo_train = coo_matrix(train_arr)
redis_conn.set('coo_train', pickle.dumps(coo_train))


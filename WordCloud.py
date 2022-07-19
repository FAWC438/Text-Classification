import matplotlib.pyplot as plt
import wordcloud  # 词云展示库
from collections import Counter
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

if __name__ == '__main__':
    TF_df = pd.read_csv('TF_Matrix.csv', index_col=0)

    wc = wordcloud.WordCloud(
        scale=16,  # 内容分辨率
        font_path='C:/Windows/Fonts/simhei.ttf',  # 设置字体格式
        background_color="white",
        max_words=200,  # 最多显示词数
        max_font_size=100  # 字体最大值
    )

    for cn, en in class_list.items():
        print('正在绘制...')
        wc.generate_from_frequencies(TF_df[en].to_dict())
        plt.imshow(wc)  # 显示词云
        plt.axis('off')  # 关闭坐标轴
        plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01, hspace=0, wspace=0)  # 调整边框
        plt.savefig('img/' + cn + '类新闻高频词云.png')
        print('完成' + cn + '类')

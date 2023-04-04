import re

import pandas as pd
from datasets import Dataset
import numpy as np
from collections import Counter
from data_select import select_questions
from data_select import select_baike_by_category
from data_select import select_baike_category


# 构建训练样本，从百科问答中随机抽取和正样本数量一样的样本作为负样本
def build_intention_samples():

    # 正样本
    positive = select_questions()
    positive = [question for _, question in positive]

    # 负样本
    negative_number = len(positive) + 500
    exclusive_categories = ['健康', '医疗健康', '育儿', '生活-美容', '生活-育儿', '生活-保健养生', '烦恼-两性问题']

    # 是否是排除类别
    def is_allow(category):
        for exclusive in exclusive_categories:
            if category.count(exclusive) > 0:
                return True
        return False

    # 获得候选类别(尽量不与问答数据集领域重叠)
    baike_categories = select_baike_category()
    baike_categories = set([category[0] for category in baike_categories])
    candidate_categories = []
    for category in baike_categories:
        if not category:
            continue
        if is_allow(category):
            continue

        candidate_categories.append(category)

    # 计算每个类别抽取多少样本
    negative_number = max(int(negative_number / len(candidate_categories)), 45)
    # 根据类别查询数据组成负样本
    negative = []
    for category in candidate_categories:
        questions = select_baike_by_category(category, negative_number)
        questions = [question[0] for question in questions]
        negative.extend(questions)


    print('正样本数量:', len(positive))
    print('负样本数量:', len(negative))
    samples = positive + negative
    print('样本总数量:', len(samples))


    # 标签
    positive_labels = np.ones(shape=[len(positive),], dtype=np.int64).tolist()
    negative_labels = np.zeros(shape=[len(negative),], dtype=np.int64).tolist()
    labels = positive_labels + negative_labels

    # 构建训练数据集，下面格式用于查看
    train_data = pd.DataFrame({'title': samples, 'label': labels})
    train_data.to_csv('data/intention.csv')

    # 构建训练数据集，下面格式用于训练
    train_data = Dataset.from_pandas(train_data)
    train_data = train_data.train_test_split(test_size=0.1)
    train_data.save_to_disk('data/intention.data')
    print('训练集分布:', Counter(train_data['train']['label']))
    print('测试集分布:', Counter(train_data['test']['label']))

    """
    正样本数量: 10000
    负样本数量: 10778
    样本总数量: 20778
    训练集分布: Counter({0: 9675, 1: 9025})
    测试集分布: Counter({0: 1103, 1: 975})
    """

if __name__ == '__main__':
    build_intention_samples()
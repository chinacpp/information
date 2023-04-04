import jieba
jieba.setLogLevel(0)
import random
import pandas as pd
import pickle
from collections import Counter
from data_select import select_questions
from data_select import select_and_show_question


# 构建问题倒排索引
def build_inverted_index():

    questions = select_questions()
    inverted_index = {}
    stopwords = [word.strip() for word in open('file/stopwords.txt')]
    for qid, question in questions:
        words = [word for word in jieba.lcut(question) if word not in stopwords]
        if len(words) == 0:
            print('分词失败问题:', question)
            continue
        # 构建索引
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = [qid]
            else:
                inverted_index[word].append(qid)

    pickle.dump(inverted_index, open('finish/keyword/inverted_index/inverted_index.pkl', 'wb'))


# 通过倒排索引返回包含关键字的候选列表
def generate_candidate(query, inversed_index, topK):

    # 输入问题分词并停用词过滤
    query = jieba.lcut(query)
    stopwords = [word.strip() for word in open('file/stopwords.txt')]
    query_words = [word for word in query if word not in stopwords]
    print('输入分词:', query_words)

    # 存储包含关键词的候选问题列表
    candidate_questions = []
    # 获得关键词对应的所有问题
    for word in query_words:
        try:
            candidate_questions.extend(inversed_index[word])
        except:
            pass
    # 选择包含关键字最多的前 100 个问题
    candidate_questions = Counter(candidate_questions).most_common(topK)
    candidate_questions = [question for question, freq in candidate_questions]

    return candidate_questions


def test():

    # 读取倒排索引
    inverted_index = pickle.load(open('finish/keyword/inverted_index/inverted_index.pkl', 'rb'))
    query_string = '宝宝的妈妈嗓子疼有点发烧孩子就是发烧'
    print('输入问题:', query_string)
    ids = generate_candidate(query_string, inverted_index, topK=10)
    print(ids)
    print('-' * 50)
    select_and_show_question(ids)


if __name__ == '__main__':
    # build_inverted_index()
    test()
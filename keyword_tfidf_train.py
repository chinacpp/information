import jieba
jieba.setLogLevel(0)
import jieba.analyse as analyse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import jieba.posseg as psg
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import torch
import torch.nn.functional as F
from data_select import select_all_questions
import re


def is_chinese_word(words):

    for word in words:
        if '\u4e00' <= word <= '\u9fff':
            continue
        else:
            return False

    return True


def cut_word(sentence):

    # n = ['n', 'nr', 'ns', 'nt', 'nl', 'nz', 'nsf', 's'] + ['v', 'vd', 'vn', 'vx'] + ['a', 'ad', 'al', 'an']

    # 粗粒度分词
    # words_with_pos = psg.cut(sentence)
    # question_words = [word for word, pos in words_with_pos if pos in p]

    # 抽取关键字
    # question_words = analyse.tfidf(sentence, allowPOS=p, topK=30)

    # 搜索引擎模式，尽可能的分出词
    question_words = jieba.lcut_for_search(sentence)
    question_words = [word for word in question_words if is_chinese_word(word)]

    # words = analyse.textrank(sentence, allowPOS=allow_pos)
    # print('同义词增强:', [synonyms.nearby(word) for word in words])

    return ' '.join(question_words)


def train_tfidf():

    questions = select_all_questions()
    questions_words = [cut_word(question) for qid, question in questions]
    max_features = 81920
    stopwords = [word.strip() for word in open('file/stopwords.txt')]
    estimator = TfidfVectorizer(max_features=max_features, stop_words=stopwords, ngram_range=(1, 2))
    estimator.fit(questions_words)

    print('特征数量:', len(estimator.get_feature_names_out()))
    print('特征内容:', estimator.get_feature_names_out()[:50])

    pickle.dump(estimator, open('finish/keyword/tfidf/tfidf.pkl', 'wb'))


if __name__ == '__main__':
    train_tfidf()
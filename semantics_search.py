import numpy as np
from gensim.models import Doc2Vec
import faiss
import pandas as pd
import torch
import torch.nn.functional as F
import jieba
jieba.setLogLevel(0)
from data_select import select_and_show_question


def search_doc2vec(question):

    # estimator1 = Doc2Vec.load('finish/semantics/doc2vec/pvdm.bin')
    # estimator2 = Doc2Vec.load('finish/semantics/doc2vec/pvdbow.bin')
    estimator = Doc2Vec.load('finish/semantics/doc2vec/pvdm.bin')
    databases = faiss.read_index('finish/semantics/doc2vec/doc2vec.faiss')

    print('输入问题:', question)

    question = jieba.lcut(question)
    # embedding1 = estimator1.infer_vector(question)
    # embedding2 = estimator2.infer_vector(question)
    embedding = estimator.infer_vector(question).tolist()
    embedding = F.normalize(torch.tensor([embedding]), dim=-1)

    distance, ids = databases.search(embedding, 5)

    questions = pd.read_csv('data/question.csv', index_col=0)
    print('关键字相似问题:')
    print(distance)
    select_and_show_question(ids[0])


def test_doc2vec():
    search_doc2vec('吃点啥药能降血压啊？')


if __name__ == '__main__':
    test_doc2vec()
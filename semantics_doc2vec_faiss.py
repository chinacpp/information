import torch
from gensim.models import Doc2Vec
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch.nn.functional as F
import jieba
jieba.setLogLevel(0)
from data_select import select_questions


def generate_question_embedding_doc2vec_single():

    estimator = Doc2Vec.load('finish/semantics/doc2vec/pvdm.bin')
    # estimator = Doc2Vec.load('finish/semantics/doc2vec/pvdbow.bin')
    questions = select_questions()

    database = faiss.IndexIDMap(faiss.IndexFlatIP(256))
    progress = tqdm(range(len(questions)), desc='开始计算问题向量')
    ids, embeddings = [], []
    for index, question in questions:
        embedding = estimator.infer_vector(jieba.lcut(question))
        # 向量转换成单位向量
        ids.append(index)
        embeddings.append(embedding.tolist())
        # 存储问题向量及其编号
        progress.update()
    progress.set_description('结束计算问题向量')
    progress.close()

    # 向量转换为单位向量
    embeddings = F.normalize(torch.tensor(embeddings), dim=-1)

    # 存储向量索引对象
    database.add_with_ids(embeddings, ids)
    faiss.write_index(database, 'finish/semantics/doc2vec/doc2vec.faiss')


# 拼接 pvdm + pvdbow 模型的向量
def generate_question_embedding_doc2vec_combine():

    estimator1 = Doc2Vec.load('finish/semantics/doc2vec/pvdm.bin')
    estimator2 = Doc2Vec.load('finish/semantics/doc2vec/pvdbow.bin')
    questions = select_questions()

    database = faiss.IndexIDMap(faiss.IndexFlatIP(256))
    progress = tqdm(range(len(questions)), desc='开始计算问题向量')
    qids, embeddings = [], []
    for qid, question in questions:

        # 拼接两个向量
        question = jieba.lcut(question)
        embedding1 = estimator1.infer_vector(question)
        embedding2 = estimator2.infer_vector(question)
        embedding = torch.concat([torch.tensor(embedding1), torch.tensor(embedding2)], dim=-1)
        qids.append(qid)
        embeddings.append(embedding)
        progress.update()
    progress.set_description('结束计算问题向量')
    progress.close()

    # 将张量列表转换为张量类型
    embeddings = torch.stack(embeddings, dim=0)
    # 向量转换为单位向量
    embeddings = F.normalize(embeddings, dim=-1)

    # 存储向量索引对象
    database.add_with_ids(embeddings, qids)
    faiss.write_index(database, 'finish/semantics/doc2vec/doc2vec.faiss')


if __name__ == '__main__':
    generate_question_embedding_doc2vec_single()
    # generate_question_embedding_doc2vec_combine()
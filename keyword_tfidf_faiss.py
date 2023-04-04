import faiss
import pickle
import numpy as np
import pandas as pd
import jieba
jieba.setLogLevel(0)
import jieba.analyse as analyse
from data_select import select_and_show_question
from data_select import select_and_show_solution
from data_select import select_questions
from keyword_tfidf_train import cut_word


def generate_tfidf_to_faiss():

    estimator = pickle.load(open('finish/keyword/tfidf/tfidf.pkl', 'rb'))
    questions = select_questions()
    questions_words = [(qid, cut_word(question)) for qid, question in questions]

    write_number = 0
    database = faiss.IndexIDMap(faiss.IndexFlatIP(81920))
    for qid, question in questions_words:
        try:  # 有些句子分出的关键词列表为空，此时跳过
            question = estimator.transform([question]).toarray().tolist()
            database.add_with_ids(np.array(question), [qid])
            write_number += 1
        except Exception as e:
            pass

    print('写入 TF-IDF 数量:', write_number)
    faiss.write_index(database, 'finish/keyword/tfidf/tfidf.faiss')


def test():

    estimator = pickle.load(open('finish/keyword/tfidf/tfidf.pkl', 'rb'))
    database = faiss.read_index('finish/keyword/tfidf/tfidf.faiss')

    # 输入问题
    # input_question = '宝宝的妈妈嗓子疼有点发烧孩子就是发烧'
    # input_question = '怀孕时乳房会有刺痛感吗'
    # input_question = '小孩发烧，吃点什么药啊？'
    # input_question = '染头发影响宝宝吃奶吗？'
    query_string = '吃点啥药能降血压啊？'
    print('输入问题:', query_string)
    query_words = [cut_word(query_string)]
    print('输入分词:', query_words)

    query_vector = estimator.transform(query_words).toarray()
    distances, ids = database.search(query_vector, 10)

    print(ids[0])
    print(distances[0].tolist())
    select_and_show_question(ids[0])
    print('-' * 100)
    select_and_show_solution(ids[0])

if __name__ == '__main__':
    # generate_tfidf_to_faiss()
    test()
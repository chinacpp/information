from transformers import logging
logging.set_verbosity_error()

import faiss
from simcse import SimCSE
from transformers import BertTokenizer
from inverted_index import generate_candidate
import torch
import pandas as pd
import pickle
from intention_predictive import intention_predict_albert
from transformers import AlbertForSequenceClassification


@torch.no_grad()
def predict():

    input_question = '我有些头疼？是不是感冒了？吃点啥药啊？'
    input_question = '脸上长了很多小疙瘩，这是过敏了吗'
    # input_question = '哈哈，我在打篮球哇'
    print('输入问题:', input_question)

    # 1. 意图识别
    model_save_path = 'albert_model/epoch_36_simcse_loss_4.5393'
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    estimator = AlbertForSequenceClassification.from_pretrained(model_save_path, num_labels=2).eval()

    result = intention_predict_albert(input_question, estimator, tokenizer)
    if not result:
        print('我不太理解您的问题，给您推荐了以下下几个问题:')
    else:
        print('根据您的问题，匹配结果为:')

    questions = pd.read_csv('data/question.csv', index_col=0)

    # 2. 倒排索引
    inversed_index = pickle.load(open('data/question_inversed_index.pkl', 'rb'))
    stopwords = [word.strip() for word in open('file/stopwords.txt')]
    candidate_list = generate_candidate(input_question, inversed_index, stopwords)
    print('关键字相似问题:')
    print(questions[questions.index.isin(candidate_list[:5])])

    print('\n')
    print('-' * 100)
    print('\n')

    # 3. 语义搜索
    model_path = 'model/epoch_40_simcse_loss_0.0506'
    estimator = SimCSE().from_pretrained(model_path).eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    databases = faiss.read_index('data/questions.index')

    input_encoding = tokenizer.encode_plus(input_question, return_token_type_ids=False, return_tensors='pt')
    input_embeddig = estimator.get_encoder_embedding(**input_encoding)
    distance, ids = databases.search(input_embeddig, 10)

    print('语义相似问题:')
    print(distance[0])
    print(questions[questions.index.isin(ids[0])])


if __name__ == '__main__':
    predict()

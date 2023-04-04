import pickle
import torch
from semantics_simcse_new import SimCSE
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import faiss
import time
from data_select import select_questions
from data_select import select_questions_by_ids


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('训练设备:', device)


def word_repetition(input_ids, dup_rate=0.32):

    # 获得 batch_size 大小，以及
    batch_size = len(input_ids)
    repetitied_input_ids = []

    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        # 统计非0的句子长度
        actual_len = np.count_nonzero(cur_input_id)
        dup_word_index = []
        # 如果句子长度大于5才进行操作
        if actual_len > 5:
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            # Skip cls and sep position
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)

        r_input_id = []
        for idx, word_id in enumerate(cur_input_id):
            # 插入重复 Token
            if idx in dup_word_index:
                r_input_id.append(word_id)
            r_input_id.append(word_id)

        repetitied_input_ids.append(r_input_id)

    # 填充补齐 batch 序列长度
    repetitied_input_ids_mask = []
    batch_maxlen = max([len(ids) for ids in repetitied_input_ids])
    for batch_id in range(batch_size):
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = batch_maxlen - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len

        mask = np.ones((len(repetitied_input_ids[batch_id]), ), dtype=np.int32)
        mask[np.array(repetitied_input_ids[batch_id]) == 0] = 0

        repetitied_input_ids_mask.append(mask.tolist())


    repetitied_input_ids = torch.tensor(repetitied_input_ids, device=device)
    repetitied_input_ids_mask = torch.tensor(repetitied_input_ids_mask, device=device)

    return repetitied_input_ids, repetitied_input_ids_mask


def test():

    batch_data = ['我肚子疼，还有头晕耳鸣，眼睛花，我该怎么办？', '眼睛睁得久了耳朵嗡嗡响，还有我的嗓子发炎了到底是怎么回事？']
    tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese')

    batch_data = tokenizer.batch_encode_plus(batch_data, return_token_type_ids=False)
    print(batch_data['input_ids'])

    print(len(batch_data['input_ids'][0]), len(batch_data['input_ids'][1]))
    a, b = word_repetition(batch_data['input_ids'])
    print(len(a[0]), len(a[1]))
    print(len(b[0]), len(b[1]))
    print(a)
    print(b)


@torch.no_grad()
def generate_negative_samples(model_path=None, epoch=0, batch_size=5, start_init=False):

    if start_init:
        estimator = SimCSE().eval().to(device)
        tokenizer = BertTokenizer.from_pretrained('pretrained/albert_chinese_tiny')
    else:
        estimator = SimCSE().from_pretrained(model_path).eval().to(device)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    questions = select_questions()

    # 对 batch 数据进行编码
    def collate_function(batch_data):
        question_index, question_input = [], []
        for index, input in batch_data:
            question_index.append(index)
            question_input.append(input)
        question_input = tokenizer(question_input, padding='longest', return_token_type_ids=False, return_tensors='pt')
        question_input = {key: value.to(device) for key, value in question_input.items()}
        return question_index, question_input

    dataloader = DataLoader(questions, batch_size=128, collate_fn=collate_function)
    progress = tqdm(range(len(dataloader)), desc='开始生成 epcoh=%d 向量' % epoch)

    qid_to_ebd = {}
    qids, ebds = [], []
    for bqid, inputs in dataloader:
        bebd = estimator.get_encoder_embedding(**inputs)
        qids.extend(bqid)
        ebds.append(bebd)
        for qid, ebd in zip(bqid, bebd):
            qid_to_ebd[qid] = ebd
        progress.update()
    progress.set_description('结束生成 epcoh=%d 向量' % epoch)
    progress.close()

    # 存储向量索引对象 [10000, 256]
    ebds = torch.concat(ebds, dim=0).cpu()
    database = faiss.IndexIDMap(faiss.IndexFlatIP(256))
    database.add_with_ids(ebds, qids)

    # 每个样本生成负样本
    _, search_ids = database.search(ebds, batch_size)
    questions = dict(questions)
    candidate_questions = []
    for qid, sqids in zip(qids, search_ids.tolist()):
        if qid in sqids:
            sqids.remove(qid)
        candidate_questions.append([questions[id] for id in [qid] + sqids[:batch_size-1]])

    return candidate_questions



def train_simcse():

    estimator = SimCSE().to(device)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)
    tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese')
    epoch_num = 20
    batch_size = 64
    dataset = select_questions()
    # 初始化负样本
    current_data = generate_negative_samples(batch_size=batch_size, start_init=True)
    current_indexes = list(range(len(current_data)))

    def collate_function(batch_data):

        ones = tokenizer(batch_data, return_token_type_ids=False)['input_ids']
        twos = tokenizer(batch_data, return_token_type_ids=False)['input_ids']

        model_inputs = {}
        input_ids, attention_mask = word_repetition(ones)
        model_inputs['query_input_ids'] = input_ids
        model_inputs['query_attention_mask'] = attention_mask

        input_ids, attention_mask = word_repetition(twos)
        model_inputs['title_input_ids'] = input_ids
        model_inputs['title_attention_mask'] = attention_mask

        return model_inputs

    for epoch in range(epoch_num):

        epoch_loss = 0.0
        random.shuffle(current_indexes)
        progress = tqdm(range(len(current_indexes)))
        for index, current_index in enumerate(current_indexes):

            model_inputs = collate_function(current_data[current_index])
            loss = estimator(**model_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += (loss.item() * len(model_inputs['query_input_ids']))
            progress.set_description('epoch %2d %8.4f' % (epoch + 1, epoch_loss))
            progress.update()
        progress.close()

        model_save_path = 'finish/semantics/simcse/train-2/%2d_semantic_simcse_loss_%.4f' % (epoch + 1, epoch_loss)
        estimator.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        current_data = generate_negative_samples(model_path=model_save_path, epoch=epoch+1, batch_size=batch_size, start_init=False)


if __name__ == '__main__':
    # test()
    train_simcse()
    # generate_negative_samples()


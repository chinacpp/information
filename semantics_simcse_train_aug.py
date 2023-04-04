import torch
from simcse import SimCSE
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


class DataSimCSE:

    def __init__(self):
        self._data = self._get_train()

    def _get_train(self):
        train = pd.read_csv('data/question.csv')['content']
        return train

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {'one': self._data[index], 'two': self._data[index]}


# 对输入进行 word repetition 增强
def collate_function_aug(tokenizer, batch_data):

    ones, twos = [], []
    for data in batch_data:
        ones.append(data['one'])
        twos.append(data['two'])

    ones = tokenizer(ones, return_token_type_ids=False)['input_ids']
    twos = tokenizer(twos, return_token_type_ids=False)['input_ids']

    model_inputs = {}
    input_ids, attention_mask = word_repetition(ones)
    model_inputs['query_input_ids'] = input_ids
    model_inputs['query_attention_mask'] = attention_mask

    input_ids, attention_mask = word_repetition(twos)
    model_inputs['title_input_ids'] = input_ids
    model_inputs['title_attention_mask'] = attention_mask

    return model_inputs


def train_simcse():

    estimator = SimCSE().to(device)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)
    tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese')
    dataloadr = DataLoader(DataSimCSE(), shuffle=True, batch_size=8, collate_fn=lambda batch_data: collate_function_aug(tokenizer, batch_data))
    epoch_num = 20

    for epoch in range(epoch_num):

        progress = tqdm(range(len(dataloadr)))
        epoch_loss = 0.0
        for model_inputs in dataloadr:
            loss = estimator(**model_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_description('epoch %2d %8.4f' % (epoch + 1, epoch_loss))
            progress.update()
        progress.close()

        if epoch > 5:
            model_save_path = 'model_aug/epoch_%d_simcse_loss_%.4f' % (epoch + 1, epoch_loss)
            estimator.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)


if __name__ == '__main__':
    # test()
    train_simcse()



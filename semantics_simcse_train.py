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


def collate_function(tokenizer, batch_data):

    ones, twos = [], []
    for data in batch_data:
        ones.append(data['one'])
        twos.append(data['two'])

    ones = tokenizer(ones, return_token_type_ids=False, padding='longest', return_tensors='pt')
    twos = tokenizer(twos, return_token_type_ids=False, padding='longest', return_tensors='pt')
    ones = {key: value.to(device) for key, value in ones.items()}
    twos = {key: value.to(device) for key, value in twos.items()}

    model_inputs = {}
    model_inputs['query_input_ids'] = ones['input_ids']
    model_inputs['title_input_ids'] = twos['input_ids']
    model_inputs['query_attention_mask'] = ones['attention_mask']
    model_inputs['title_attention_mask'] = twos['attention_mask']

    return model_inputs


def train_simcse():

    estimator = SimCSE().to(device)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)
    tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese')
    dataloadr = DataLoader(DataSimCSE(), shuffle=True, batch_size=8, collate_fn=lambda batch_data: collate_function(tokenizer, batch_data))
    epoch_num = 40

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

        model_save_path = 'model/epoch_%d_simcse_loss_%.4f' % (epoch + 1, epoch_loss)
        estimator.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)


if __name__ == '__main__':
    train_simcse()



import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AlbertModel
import torch


class SentenceBert(nn.Module):

    def __init__(self, model_path='model/albert_chinese_tiny'):
        super(SentenceBert, self).__init__()
        self.basemodel = AlbertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)


    # 输入: [(sentence1, sentence2), (sentence1, sentence2)...]
    def forward(self, batch_sentence_pairs):

        # 1. 将 sentence1 放到第一个列表中，sentence2 放到第二个列表中
        batch_sentence = [[], []]
        for sentence1, sentence2 in batch_sentence_pairs:
            batch_sentence[0].append(sentence1)
            batch_sentence[1].append(sentence2)

        def batch_encode_plus(sentences):
            batch_inputs = self.tokenizer.batch_encode_plus(sentences,
                                                            padding='longest',
                                                            return_tensors='pt')
            # 将张量移动到 device 计算设备
            batch_inputs = { key: value.to(device) for key, value in batch_inputs.items()}

            return batch_inputs

        # 2. 分别计算第一个、第二个列表中所有句子的编码
        self.basemodel.train()
        batch_encodes = [batch_encode_plus(sentence) for sentence in batch_sentence]
        batch_outputs = [self.basemodel(**sentence_encode) for sentence_encode in batch_encodes]

        # 3. 计算每个句子的向量表示，这里使用平均 token 向量的方式表示 sentence 向量
        sentence_embeddings = []
        for index, outputs in enumerate(batch_outputs):
            token_embd = outputs.last_hidden_state
            token_mask = batch_encodes[index]['attention_mask'].unsqueeze(-1).expand(token_embd.size())
            sentence_embedding = torch.sum(token_embd * token_mask, 1) / torch.sum(token_mask, 1)
            sentence_embeddings.append(sentence_embedding)

        # 4. 计算 sentence1 和 sentence2 的相似度
        similarities = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])

        return similarities


    def save(self, model_path_name):
        self.basemodel.save_pretrained(model_path_name)
        self.tokenizer.save_pretrained(model_path_name)


    def encode(self, sentence):

        self.basemodel.eval()
        sentence_encode = self.tokenizer.encode_plus(sentence, return_tensors='pt')
        with torch.no_grad():
            sentence_output = self.basemodel(**sentence_encode)
        token_embd = sentence_output.last_hidden_state
        embedding = torch.sum(token_embd, dim=1) / token_embd.shape[1]

        return embedding.squeeze()
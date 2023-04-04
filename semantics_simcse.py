import torch
import torch.nn as nn
import torch.nn.functional as F



class SimCSE(nn.Module):

    def __init__(self, dropout=0.1, margin=0.0, scale=20, ebdsize=256):
        """
        :param dropout: 丢弃率
        :param margin: 当两个样本的句子小于 margin 则认为是相似的
        :param scale: 放大相似度的值，帮助模型更好地区分不同的相似度级别，有利于模型收敛
        :param ebdsize: 输出的向量维度
        """

        super(SimCSE, self).__init__()
        # 初始化编码器对象
        # from transformers import BertModel
        # self.encoder = BertModel.from_pretrained('pretrained/bert-base-chinese')
        # self.encoder = BertModel.from_pretrained('pretrained/chinese-bert-wwm-ext')
        from transformers import AlbertModel
        self.encoder = AlbertModel.from_pretrained('pretrained/albert_chinese_tiny')
        # 设置随机丢弃率
        self.dropout = nn.Dropout(dropout)
        # 控制输出向量维度
        self.ebdsize = ebdsize
        self.outputs = nn.Linear(self.encoder.config.hidden_size, ebdsize)

        self.margin = margin
        self.sacle = scale

    def get_encoder_embedding(self, input_ids, attention_mask=None, with_pooler=False):

        # 输出的结果经过了池化
        sequence_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if with_pooler:
            cls_embedding = sequence_output.pooler_output
        else:
            cls_embedding = sequence_output.last_hidden_state[:, 0, :]

        cls_embedding = self.outputs(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        # 对向量进行 normalize，即每个向量除以其二范数(单位向量)，计算余弦相似度时，只需要进行点积计算即可
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)

        return cls_embedding


    def save_pretrained(self, model_save_path):

        self.encoder.save_pretrained(model_save_path)
        torch.save(self.outputs.state_dict(), model_save_path + '/dim_reduce.pth')
        model_param = {'dropout': self.dropout, 'ebdsize': self.ebdsize, 'margin': self.margin, 'sacle': self.sacle}
        torch.save(model_param, model_save_path + '/model_param.pth')


    def from_pretrained(self, model_save_path):

        model_param = torch.load(model_save_path + '/model_param.pth')
        self.sacle = model_param['sacle']
        self.margin = model_param['margin']
        self.dropout = model_param['dropout']
        self.ebdsize = model_param['ebdsize']
        self.encoder.from_pretrained(model_save_path)
        self.outputs.load_state_dict(torch.load(model_save_path + '/dim_reduce.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        return self


    def forward(self, query_input_ids, title_input_ids, query_attention_mask=None, title_attention_mask=None):

        query_cls_embedding = self.get_encoder_embedding(query_input_ids, query_attention_mask)
        title_cls_embedding = self.get_encoder_embedding(title_input_ids, title_attention_mask)

        # 获得输入数据的计算设备
        device = query_cls_embedding.device

        # query_cls_embedding 和 title_cls_embedding 已经经过了标准化，此处直接相乘得到余弦相似度
        # 计算两两句子的相似度
        # torch.Size([3, 256])
        # torch.Size([256, 3])
        cosine_sim = torch.matmul(query_cls_embedding, title_cls_embedding.transpose(1, 0))

        # tensor([0., 0., 0.])
        margin_diag = torch.full(size=[query_cls_embedding.size(0)], fill_value=self.margin).to(device)

        # tensor([[0.8873, 0.8727, 0.8366],
        #         [0.8876, 0.8834, 0.9100],
        #         [0.9068, 0.9079, 0.8703]], grad_fn=<SubBackward0>)
        cosine_sim = cosine_sim - torch.diag(margin_diag)

        # 放大相似度，有利于模型收敛
        # tensor([[17.7461, 17.4537, 16.7329],
        #         [17.7512, 17.6680, 18.2001],
        #         [18.1369, 18.1573, 17.4062]], grad_fn=<MulBackward0>)
        cosine_sim *= self.sacle

        # 构建标签
        # tensor([0, 1, 2])
        labels = torch.arange(0, query_cls_embedding.size(0)).to(device)

        # 计算损失
        # tensor(1.2422, grad_fn=<NllLossBackward0>)
        loss = F.cross_entropy(input=cosine_sim, target=labels)


        return loss

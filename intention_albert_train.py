from transformers import AlbertForSequenceClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from datasets import load_from_disk
import torch.optim as optim
import torch.nn as nn
import glob
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


# 计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(batch_data, tokenizer):

    titles, labels = [], []
    for data in batch_data:
        titles.append(data['title'])
        labels.append(data['label'])

    title_tensor = tokenizer.batch_encode_plus(titles,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
    title_tensor = {key: value.to(device) for key, value in title_tensor.items()}
    label_tensor = torch.tensor(labels, device=device)
    return title_tensor, label_tensor


def train_albert():

    # https://huggingface.co/clue/albert_chinese_tiny
    estimator = AlbertForSequenceClassification.from_pretrained('pretrained/albert_chinese_tiny', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('pretrained/albert_chinese_tiny')
    traindata = load_from_disk('data/intention.data')['train']
    dataloader = DataLoader(traindata, batch_size=128, shuffle=True, collate_fn=lambda data: collate_function(data, tokenizer))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.6, patience=2, cooldown=2, verbose=True)

    for epoch in range(30):

        total_loss, total_size, total_corr = 0.0, 0, 0
        progress = tqdm(range(len(dataloader)))
        for title_tensor, label_tensor in dataloader:

            outputs = estimator(**title_tensor)
            loss = criterion(outputs.logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 预测标签
            y_pred = torch.argmax(outputs.logits, dim=-1)
            total_corr += (y_pred == label_tensor).sum().item()
            total_loss += loss.item() * len(label_tensor)
            total_size += len(label_tensor)

            # 更新进度
            desc = '%2d. %6.1f %5d/%5d %.4f %.2E' % (epoch + 1, total_loss, total_corr, total_size, total_corr/total_size, scheduler.optimizer.param_groups[0]['lr'])
            progress.set_description(desc)
            progress.update()

        scheduler.step(total_loss)
        progress.close()

        if epoch > 5:
            model_save_path = 'finish/intention/albert/%0d_intention_albert_loss_%.4f' % (epoch + 1, total_loss)
            estimator.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)


@torch.no_grad()
def eval_model(model_name):

    estimator = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    traindata = load_from_disk('data/intention.data')['test']
    dataloader = DataLoader(traindata, batch_size=128, shuffle=True, collate_fn=lambda data: collate_function(data, tokenizer))

    model_name = model_name[model_name.rfind('/') + 1:]
    progress = tqdm(range(len(dataloader)), desc='%30s' % model_name)
    y_true, y_pred = [], []
    for inputs_tensor, labels_tensor in dataloader:
        outputs = estimator(**inputs_tensor)
        y_label = torch.argmax(outputs.logits, dim=-1)
        y_pred.extend(y_label.cpu().numpy().tolist())
        y_true.extend(labels_tensor.cpu().numpy().tolist())
        progress.update()
    progress.close()


    print('准确率:', accuracy_score(y_true, y_pred))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)
    print('-' * 100)


def eval_albert():

    model_names = glob.glob('finish/intention/albert/*intention_albert*')
    for model_name in model_names:
        eval_model(model_name)

def predict(inputs):
    model_save_path = 'albert_model/epoch_36_simcse_loss_4.5393'
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    estimator = AlbertForSequenceClassification.from_pretrained(model_save_path, num_labels=2).eval()

    # 对输入处理编码
    inputs = tokenizer.encode_plus(inputs,
                                   return_token_type_ids=False,
                                   return_attention_mask=False,
                                   return_tensors='pt')
    # 模型预测
    with torch.no_grad():
        outputs = estimator(**inputs)
        y_pred = torch.argmax(outputs.logits)

        if y_pred.item() == 1:
            print('\033[31m需要处理的问题\033[m')
        else:
            print('其他方面的问题')

def test():
    predict('为什么抽完血后会出现头晕、四肢无力脸色发白冒汗等现象？')
    predict('我是怀孕了吗')
    predict('嗓子起疮，这是什么原因导致的？')
    predict('哈哈')
    predict('我滴妈呀，你真笨啊')
    predict('前天早上在医院的广场上玩篮球，一会来了几个病人，我们就一起玩了')
    predict('我是医院的病人，我发烧了，所以在这里住院')


if __name__ == '__main__':
    # train_albert()
    eval_albert()
    # test()
import torch
from transformers import AlbertForSequenceClassification
from transformers import BertTokenizer
from pyhanlp import JClass


@torch.no_grad()
def intention_predict_albert(question, model, tokenizer):

    # 规范化输入文本
    normalizer = JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    question = normalizer.convert(question)
    # 对输入文本编码
    question = tokenizer.encode_plus(question, return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')

    # 模型预测问题类别
    outputs = model(**question)
    y_pred = torch.argmax(outputs.logits)

    return y_pred.item() == 1



def test_albert():

    model_save_path = 'albert_model/epoch_36_simcse_loss_4.5393'
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    estimator = AlbertForSequenceClassification.from_pretrained(model_save_path, num_labels=2)
    estimator.eval()

    ret = intention_predict_albert('我肚子疼，还发着高烧，还有头晕耳鸣，眼睛花，我该怎么办？', estimator, tokenizer)
    print(ret)
    ret = intention_predict_albert('我前天去操场打球了', estimator, tokenizer)
    print(ret)


if __name__ == '__main__':
    test_albert()


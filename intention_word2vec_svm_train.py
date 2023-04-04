import pickle

from sklearn.svm import SVC
from datasets import load_from_disk
import jieba
jieba.setLogLevel(0)
import fasttext
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import datasets
datasets.disable_progress_bar()


def train_svm():

    traindata = load_from_disk('data/intention.data')['train']
    tokenizer = fasttext.load_model('pretrained/cc.zh.300.bin')

    def collate_function(batch_data):
        titles = batch_data['title']
        labels = batch_data['label']
        model_inputs = []
        for title in titles:
            inputs = tokenizer.get_sentence_vector(' '.join(jieba.lcut(title)))
            model_inputs.append(inputs.tolist())
        return {'title': model_inputs, 'label': labels}

    # 数据向量化
    traindata = traindata.map(collate_function, batched=True, batch_size=32)
    # 训练支持向量机
    estimator = SVC()
    estimator.fit(traindata['title'], traindata['label'])
    # 存储模型
    pickle.dump(estimator, open('finish/intention/svm/svm.pkl', 'wb'))


def eval_svm():

    estimator = pickle.load(open('finish/intention/svm/svm.pkl', 'rb'))
    tokenizer = fasttext.load_model('pretrained/cc.zh.300.bin')

    traindata = load_from_disk('data/intention.data')
    def collate_function(batch_data):
        titles = batch_data['title']
        labels = batch_data['label']
        model_inputs = []
        for title in titles:
            inputs = tokenizer.get_sentence_vector(' '.join(jieba.lcut(title)))
            model_inputs.append(inputs.tolist())
        return {'title': model_inputs, 'label': labels}
    traindata = traindata.map(collate_function, batched=True, batch_size=32)

    # 训练集准确率
    y_pred = estimator.predict(traindata['train']['title'])
    y_true = traindata['train']['label']
    print('准确率:', accuracy_score(y_true, y_pred))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)
    print('-' * 50)

    # 测试集准确率
    y_pred = estimator.predict(traindata['test']['title'])
    y_true = traindata['test']['label']
    print('测试集:', accuracy_score(y_true, y_pred))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)

"""
Parameter 'function'=<function train_svm.<locals>.collate_function at 0x7f1ef3542170> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
训练集: 0.9879503698401336
(array([0.99103551, 0.98462111]), array([0.98582371, 0.99027067]), array([0.98842274, 0.98743781]), array([8747, 8017]))
测试集: 0.9880725190839694
(array([0.98912551, 0.98690176]), array([0.98822997, 0.98789713]), array([0.98867754, 0.98739919]), array([2209, 1983]))
"""

if __name__ == '__main__':
    train_svm()
    eval_svm()



from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
import jieba
jieba.setLogLevel(0)
from data_select import select_questions


class Callback(CallbackAny2Vec):

    def __init__(self, epoch):
        super(Callback, self).__init__()
        self.progress = tqdm(range(epoch))

    def on_train_begin(self, model):
        self.progress.set_description('开始句向量训练')

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        self.progress.set_description('正在句向量训练')
        self.progress.update()

    def on_train_end(self, model):
        self.progress.set_description('结束句向量训练')
        self.progress.close()



# 语料由所有的问题和答案组成
def build_train_corpus():

    # 读取原问题、增强问题、问题答案
    questions = select_questions()

    # 对文档进行标记
    model_train_corpus = []
    for qid, question in questions:
        tokens = jieba.lcut(question)
        # TaggedDocument 用于构建 doc2vec 的训练语料，第二个参数用于标记样本唯一性
        document = TaggedDocument(tokens, [qid])
        model_train_corpus.append(document)

    return model_train_corpus


epochs = 100

def train_doc2vec_pvdm():

    # 构建训练语料
    documents = build_train_corpus()
    # 训练向量模型
    model = Doc2Vec(vector_size=256, negative=128, alpha=1e-4, dm=1, min_count=2, epochs=epochs, window=5, seed=42)
    # 构建训练词表
    model.build_vocab(documents)
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=model.epochs,
                # start_alpha=1e-2,
                # end_alpha=1e-4,
                callbacks=[Callback(epochs)])

    # 存储词向量模型
    model.save('finish/semantics/doc2vec/pvdm.bin')



def train_doc2vec_pvbbow():

    # 构建训练语料
    documents = build_train_corpus()
    # 训练向量模型
    model = Doc2Vec(vector_size=128, negative=5, alpha=1e-4, dm=0, min_count=2, epochs=epochs, window=5, seed=42)
    # 构建训练词表
    model.build_vocab(documents)
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=model.epochs,
                # start_alpha=1e-2,
                # end_alpha=1e-4,
                callbacks=[Callback(epochs)])

    # 存储词向量模型
    model.save('finish/semantics/doc2vec/pvdbow.bin')


if __name__ == '__main__':
    train_doc2vec_pvdm()
    # train_doc2vec_pvbbow()

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pickle
import jieba.posseg as psg
import jieba
jieba.setLogLevel(0)


def cut_word(sentence):

    allow = ['n', 'nr', 'ns', 'nt', 'nl', 'nz', 'nsf', 's'] + ['v', 'vd', 'vn', 'vx'] + ['a', 'ad', 'al', 'an']
    stopwords = [word.strip() for word in open('file/stopwords.txt')]
    sentence_words = []
    sentence = psg.lcut(sentence)
    for word, pos in sentence:
        if pos not in allow:
            continue
        if word in stopwords:
            continue
        sentence_words.append(word)

    return ' '.join(sentence_words)


def train_vectorizer():

    questions = load_from_disk('data/intention.data')['train']
    questions = [cut_word(question) for question in questions['title']]
    tokenizer = CountVectorizer(max_features=21246)
    tokenizer.fit(questions)
    print('特征数:', len(tokenizer.get_feature_names_out()))
    pickle.dump(tokenizer, open('finish/intention/bayes/vectorizer.pkl', 'wb'))


def train_bayes_model():

    vectorizer = pickle.load(open('finish/intention/bayes/vectorizer.pkl', 'rb'))
    questions = load_from_disk('data/intention.data')['train']
    inputs = [cut_word(title) for title in questions['title']]
    labels = questions['label']
    inputs = vectorizer.transform(inputs)
    estimator = MultinomialNB()
    estimator.fit(inputs, labels)
    pickle.dump(estimator, open('finish/intention/bayes/bayes.pkl', 'wb'))


def eval_bayes_model():

    vectorizer = pickle.load(open('finish/intention/bayes/vectorizer.pkl', 'rb'))
    estimator = pickle.load(open('finish/intention/bayes/bayes.pkl', 'rb'))
    questions = load_from_disk('data/intention.data')['test']
    inputs = [cut_word(question) for question in questions['title']]
    labels = questions['label']
    inputs = vectorizer.transform(inputs)
    ypreds = estimator.predict(inputs)

    precision, recall, f_score, true_sum = precision_recall_fscore_support(labels, ypreds)
    print('准确率:', accuracy_score(labels, ypreds))
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)

    """
    准确率: 0.9701636188642926
    精确率: [0.99430199 0.94536585]
    召回率: [0.94922937 0.99384615]
    F-score: [0.97124304 0.969     ]
    """

if __name__ == '__main__':
    train_vectorizer()
    train_bayes_model()
    eval_bayes_model()
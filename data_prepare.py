import numpy as np
import pandas as pd
import codecs
import re
from string import punctuation


def clean_sentence(sentence):

    sentence = sentence.strip()
    sentence = re.sub(r'[\"\']', '', sentence)
    sentence = re.sub(r'[“”※★]', '', sentence)

    # 去掉开头的标点符号
    start = 0
    not_allowed = list(punctuation) + ['、', '~', '`', '・', '。', '，', '！', '￥', '……', '（', '）', '「', '」', '；', '？']
    for start, word in enumerate(sentence):
        if word not in not_allowed:
            break
    sentence = sentence[start:]

    # 去掉结尾的标点符号(除了句号、问号、省略号、感叹号)
    end = len(sentence) - 1
    not_allowed = ['"', '~', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', '：', ';', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '、', '・', '，','￥', '（', '）', '「', '」', '；']
    for end in range(len(sentence) - 1, -1, -1):
        if sentence[end] not in not_allowed:
            break
    sentence = sentence[:end + 1]

    return sentence


def prepare_all_data():

    # 1. 读取 question
    file = codecs.open('cMedQA_v2.0/question.csv')
    file.readline()  # 跳过表头行
    questions = {'qid': [], 'question': []}
    # 将原始的qid映射为从0开始的qid
    qid_to_nid = {}
    for nid, line in enumerate(file):
        qid, question = line.split(',')
        question = clean_sentence(question)
        questions['qid'].append(nid)
        questions['question'].append(question)
        qid_to_nid[qid] = nid

    file.close()
    print('问题数量:', len(questions['qid']))

    # 2. 读取 solution
    file = codecs.open('cMedQA_v2.0/solution.csv')
    file.readline()
    solutions = {'sid': [], 'qid': [], 'solution': []}
    for sid, line in enumerate(file):
        _, qid, solution = line.split(',')
        if qid not in qid_to_nid:
            continue
        solution = clean_sentence(solution)
        solutions['sid'].append(sid)
        solutions['qid'].append(qid_to_nid[qid])
        solutions['solution'].append(solution)
    file.close()
    print('答案数量:', len(solutions['qid']))

    # 3. 存储 quetions 和 solution
    questions = pd.DataFrame(questions).drop_duplicates(['question'])
    questions.to_csv('data/question_all.csv')
    solutions = pd.DataFrame(solutions).drop_duplicates(['solution'])
    solutions.to_csv('data/solution_all.csv')

    print('----去重之后----')
    print('问题数量:', len(questions))
    print('答案数量:', len(solutions))



def prepare_train_data():

    train_number = 10000
    # 1. 读取 question
    file = codecs.open('data/question_all.csv')
    file.readline()  # 跳过表头行
    questions = {'qid': [], 'question': []}
    for line in file:
        _, qid, question = line.split(',')
        questions['qid'].append(qid)
        questions['question'].append(question.strip())
        if len(questions['qid']) == train_number:
            break
    file.close()
    print('问题数量:', len(questions['qid']))

    # 2. 读取 solution
    file = codecs.open('data/solution_all.csv')
    file.readline()
    solutions = {'sid': [], 'qid': [], 'solution': []}
    for line in file:
        p, sid, qid, solution = line.split(',')
        if qid not in questions['qid']:
            continue
        solutions['sid'].append(sid)
        solutions['qid'].append(qid)
        solutions['solution'].append(solution.strip())
    file.close()
    print('答案数量:', len(solutions['qid']))

    # 3. 存储 quetions 和 solution
    pd.DataFrame(questions).to_csv('data/question.csv')
    pd.DataFrame(solutions).to_csv('data/solution.csv')


def prepare_baike_data():


    def short_name(cag_name):
        start = 0
        for index in range(len(cag_name)-1, -1, -1):
            if cag_name[index] == '-' or cag_name[index] == '/':
                start = index
                break
        cag_name = cag_name[start + 1:]
        return cag_name


    baike = pd.read_json('file/baiketrain.json', lines=True)
    baike = baike[['title', 'category']]
    print(baike.shape)
    baike = baike.drop_duplicates(['title'])
    print(baike.shape)

    def condition(title):
        words = [word for word in title if '\u4e00' <= word <='\u9fa5']
        if len(words) < 6:
            return True
        if len(words) / len(title) < 0.7:
            return True
        return False

    is_del = [condition(title) for title, category in baike.to_numpy()]
    baike = baike[np.array(is_del) == False]
    print(baike.shape)
    baike['title'] = [clean_sentence(title) for title in baike['title']]
    baike.to_csv('data/baike.csv')


    """
    问题数量: 120000
    答案数量: 226266
    ----去重之后----
    问题数量: 119379
    答案数量: 196571
    
    问题数量: 10000
    答案数量: 16463
    """

if __name__ == '__main__':
    prepare_all_data()
    # print('-' * 50)
    prepare_train_data()
    # prepare_baike_data()
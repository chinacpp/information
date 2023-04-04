import pickle
import faiss
from semantics_simcse import SimCSE
from tqdm import tqdm
import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from data_select import select_questions
from data_select import select_and_show_question
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


model_path = 'finish/semantics/simcse/train-1/epoch_10_simcse_loss_0.0003'


@torch.no_grad()
def generate_question_embedding_simcse():

    estimator = SimCSE().from_pretrained(model_path).eval().to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    questions = select_questions()

    def collate_function(batch_data):

        question_index = []
        question_input = []
        for index, input in batch_data:
            question_index.append(index)
            question_input.append(input)

        question_input = tokenizer.batch_encode_plus(question_input,
                                                     padding='longest',
                                                     return_token_type_ids=False,
                                                     return_tensors='pt')
        question_input = {key: value.to(device) for key, value in question_input.items()}
        return question_index, question_input

    dataloadr = DataLoader(questions, batch_size=8, collate_fn=collate_function)
    progress = tqdm(range(len(dataloadr)))

    database = faiss.IndexIDMap(faiss.IndexFlatIP(256))

    for question_index, question_input in dataloadr:
        embeddings = estimator.get_encoder_embedding(**question_input)
        # 存储问题向量及其编号
        database.add_with_ids(embeddings.squeeze().cpu().numpy(), question_index)
        progress.update()
    progress.close()

    # 存储向量索引对象
    faiss.write_index(database, 'finish/semantics/simcse/simcse.faiss')


@torch.no_grad()
def test_simcse():

    estimator = SimCSE().from_pretrained(model_path).eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    databases = faiss.read_index('finish/semantics/simcse/simcse.faiss')

    # input_question = '怀孕时乳房会有刺痛感吗'
    # input_question = '小孩发烧，吃点什么药啊？'
    input_question = '染头发影响宝宝吃奶吗？'
    # input_question = '吃点啥药能降血压啊？'
    # input_question = '每次去医院，看到医生拿着针管，我就害怕的不行，只要一抽血，我就哆嗦'
    input_encoding = tokenizer.encode_plus(input_question, return_token_type_ids=False, return_tensors='pt')
    input_embeddig = estimator.get_encoder_embedding(**input_encoding)
    # 查找相似向量
    distance, ids = databases.search(input_embeddig, 10)
    print(distance)
    select_and_show_question(ids[0])


if __name__ == '__main__':
    # generate_question_embedding_simcse()
    test_simcse()



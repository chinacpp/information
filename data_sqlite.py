from pymysql import connect
import pandas as pd
import sys
import sqlalchemy
import sqlite3
import os


HOST = '192.168.2.101'
USER = 'root'
PASSWORD = '332572'
DB_NAME = 'cmed'
QUESTION_TABLE_NAME = 'questions'
SOLUTION_TABLE_NAME = 'solutions'
QUESTION_TABLE_NAME_ALL = 'questions_all'
SOLUTION_TABLE_NAME_ALL = 'solutions_all'
BAIKE_TABLE_NAME = 'baike'


def data_to_mysql():

    db_path = f'finish/{DB_NAME}.db'

    if os.path.exists(db_path):
        os.remove(db_path)
        print('数据库已存在，先删除，重建新库...')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except:
        print('数据库连接失败')
        return

    # 创建数据库表
    question_sql = f'create table {QUESTION_TABLE_NAME}(qid INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL);'
    solution_sql = f'create table {SOLUTION_TABLE_NAME}(sid INTEGER PRIMARY KEY AUTOINCREMENT, solution TEXT NOT NULL, qid int not null, FOREIGN KEY(qid) REFERENCES {QUESTION_TABLE_NAME}(qid));'

    try:
        cursor.execute(question_sql)
        cursor.execute(solution_sql)
    except Exception as e:
        print(f'创建数据表 {QUESTION_TABLE_NAME}、{SOLUTION_TABLE_NAME} 失败!')
        print(e)
        return


    engine = sqlalchemy.create_engine('sqlite:///' + db_path)

    try:
        questions = pd.read_csv('data/question.csv', index_col=0)
        questions.to_sql(name=QUESTION_TABLE_NAME, con=conn, if_exists='replace', index=False)

        solutions = pd.read_csv('data/solution.csv', index_col=0)
        solutions.to_sql(name=SOLUTION_TABLE_NAME, con=conn, if_exists='replace', index=False)

        questions_all = pd.read_csv('data/question_all.csv', index_col=0)
        questions_all.to_sql(name=QUESTION_TABLE_NAME_ALL, con=conn, if_exists='replace', index=False)

        solutions_all = pd.read_csv('data/solution_all.csv', index_col=0)
        solutions_all.to_sql(name=SOLUTION_TABLE_NAME_ALL, con=conn, if_exists='replace', index=False)

        baike = pd.read_csv('data/baike.csv', index_col=0)
        baike.to_sql(name=BAIKE_TABLE_NAME, con=conn, if_exists='replace', index=False)

    except Exception as e:
        print('数据插入失败')
        print(e)
        return

    cursor.close()
    conn.close()


if __name__ == '__main__':
    data_to_mysql()
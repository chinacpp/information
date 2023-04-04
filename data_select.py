import sqlite3


db_path = 'finish/cmed.db'


def select_baike_by_category(category, limit):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f'SELECT title FROM baike WHERE category="{category}" limit {limit}'
    cursor.execute(sql)
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results


def select_baike_category():

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT category FROM baike')
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results


def select_questions():

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM questions')
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results


def select_all_questions():

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM questions_all')
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results


def select_all_solutions():

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM solutions_all')
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results


def select_solutions():

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM solutions')
    query_results = cursor.fetchall()
    cursor.close()
    conn.close()
    return query_results



def select_questions_by_ids(qids):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f'SELECT * FROM questions WHERE qid in {str(tuple(qids))}'
    cursor.execute(sql)
    query_results = cursor.fetchall()

    # 结果按照输入id顺序排列
    query_results = dict(query_results)
    query_results = [(qid, query_results[qid]) for qid in qids]
    cursor.close()
    conn.close()
    return query_results


def select_solutions_by_ids(qids):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f'select * from solutions where qid in {str(tuple(qids))}'
    cursor.execute(sql)
    query_results = cursor.fetchall()

    print(query_results)

    # 结果按照输入id顺序排列
    query_results = { qid: solution for sid, qid, solution in query_results}
    query_results = [(qid, query_results[qid]) for qid in qids]

    cursor.close()
    conn.close()
    return query_results


def select_and_show_question(qids):
    questions = select_questions_by_ids(qids)
    for qid, quetion in questions:
        print(qid, quetion)


def select_and_show_solution(qids):
    solutions = select_solutions_by_ids(qids)
    for qid, solution in solutions:
        print(qid, solution)


if __name__ == '__main__':
    select_questions([1, 3, 5])
    select_solutions([1, 3, 5])
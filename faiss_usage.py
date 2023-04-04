import faiss
import numpy as np
import torch
import torch.nn.functional as F


# 生成浮点数向量
def generate_data():

    torch.manual_seed(0)
    # 构建输入向量
    data = torch.randn(10000, 256)
    # 向量标准化，计算点积相似度相当于计算余弦相似度
    data = F.normalize(data, dim=-1)
    # 生成向量编号
    ids = torch.arange(10000)

    return ids, data


# 存储浮点数向量
def demo01():

    # 生成数据
    ids, data = generate_data()

    # 初始化 faiss 向量索引
    dim, nlist = 256, 100

    quantizer = faiss.IndexFlatIP(dim)
    # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    # index.train(data)
    index = faiss.IndexIDMap(quantizer)

    # 向量和编号存储到索引对象中
    index.add_with_ids(data, ids)

    # 搜索最相似的N个向量
    distance, id = index.search(data[:2], 2)
    print('相似ID:\n', id)
    print('相似度:\n', distance)

    # 持久化索引对象
    faiss.write_index(index, './vector.index')


# 读取浮点数向量
def demo02():

    # 生成数据
    ids, data = generate_data()
    # 读取索引
    index = faiss.read_index('./vector.index')
    # 搜索相似向量
    distance, id = index.search(data[:2], 2)
    print('相似ID:\n', id)
    print('相似度:\n', distance)


# 存储二进制向量
def demo03():
    import numpy as np
    import faiss

    # 创建索引
    d = 32  # 向量维度
    index = faiss.IndexBinaryFlat(d)

    # 创建一些带有标识符的二进制向量
    n = 1000
    x = np.random.randint(2, size=(n, d)).astype('uint8')
    ids = np.arange(n).astype('int64')

    # 将向量添加到索引中
    index.add_with_ids(x, ids)

    # 搜索最近邻
    k = 5  # 每个查询向量的最近邻数
    xq = np.random.randint(2, size=(1, d)).astype('uint8')  # 查询向量
    D, I = index.search(xq, k)

    # 打印结果
    print('查询向量: ', xq)
    print('最近邻距离: ', D)
    print('最近邻索引: ', I)


if __name__ == '__main__':
    # demo01()
    # print('-' * 100)
    # demo02()
    # print('-' * 100)
    demo03()













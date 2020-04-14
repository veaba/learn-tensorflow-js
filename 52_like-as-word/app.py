import jieba
from gensim import corpora, models, similarities  # 文本相似度库gensim
import pandas as pd

# 三个空格隔开,才能被pd.read_csv读取到
dataset_path = "./train-data.txt"

# 列名
column_names = ["行数", "段落"]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values=" ", comment="\n",
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
# print("全部数据===>\n", dataset)
print("dataset.tail()==>\n", dataset.tail())  # 查看尾部的后五条数据
# print("数据清洗==>\n", dataset.isna().sum())  # 不懂
dataset.isna().sum()
# 拆分数据集和测试集

train_dataset = dataset.sample(frac=0.8, random_state=0)  # 8/10 训练集
test_dataset = dataset.drop(train_dataset.index)  # 2/10 测试集

print("数据集===>\n", train_dataset)
print("测试集===>\n", test_dataset)

print("行数===>\n", train_dataset["行数"])

row_data = train_dataset["行数"]
paragraph_data = train_dataset["段落"]
for i in row_data:
    print("i===>", i)

print("行数的数组===>\n", list(row_data))
print("段落的数组===>\n", list(paragraph_data))

# todo 未完待续
# # 对目标文档分词，保存在all_doc_list
# train_docs_split = []
# test_docs_split = []
# # 训练文档分词，保存在train_docs_list
# for doc in train_docs_list:
#     doc_list = [word for word in jieba.cut(doc)]
#     train_docs_split.append(doc_list)
# print("分词结果=>", train_docs_split)
#
# # 测试文档分词，保存在test_docs_list
# for doc in test_docs_list:
#     doc_list = [word for word in jieba.cut(doc)]
#     test_docs_split.append(doc_list)
#
# # 制作语料库，词袋 bag-of-words
# dict_lib = corpora.Dictionary(train_docs_split)
# print("打印一下语料库=>", dict_lib)
# print("词用数字进行编号关系=>", dict_lib.keys())
# print("编号与词的对应关系=>", dict_lib.token2id)
#
# # 使用doc2bow制作语料库
# corpus = [dict_lib.doc2bow(doc) for doc in train_docs_split]
# print("使用doc2bow制作训练语料库=>", corpus)
#
# for doc in test_docs_split:
#
#     print("测试doc==>",doc)
#     docs_test_vec = dict_lib.doc2bow(doc)
#     print("使用doc2bow制作测试语料库=>", docs_test_vec)
#
#     # 相似度分析，使用TF-IDF模型对语料库进行建模
#     tfidf = models.TfidfModel(corpus)
#
#     # 获取测试文档中，每个词的TF-IDF值
#     print("测试文档中TF-IDF值=>", tfidf[docs_test_vec])
#
#     # 对每个目标文档，分析测试文档的相似度
#     index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dict_lib.keys()))
#     sim = index[tfidf[docs_test_vec]]
#     print("分析测试文档的相似度=>", sim)
#
#     # 根据相似度排序
#     print("根据相似度排序=>", sorted(enumerate(sim), key=lambda item: -item[1]))

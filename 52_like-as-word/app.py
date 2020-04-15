import jieba
from gensim import corpora, models, similarities  # 文本相似度库gensim
import pandas as pd

# 三个空格隔开,才能被pd.read_csv读取到
dataset_train_path = "./train-data.txt"
dataset_test_path = "./test-data.txt"

# 索引值所对应的行数
row_json_map = {
    "0": "一",
    "1": "二",
    "2": "三",
    "3": "四",
    "4": "五",
    "5": "六",
    "6": "七",
    "7": "八",
    "8": "九",
    "9": "十",
}
# 列名
column_names = ["段落"]

raw_train_dataset = pd.read_csv(dataset_train_path, names=column_names,
                                na_values=" ", comment="\n",
                                sep=" ", skipinitialspace=True)
raw_test_dataset = pd.read_csv(dataset_test_path, names=column_names,
                               na_values=" ", comment="\n",
                               sep=" ", skipinitialspace=True)
train_dataset = raw_train_dataset.copy()
test_dataset = raw_test_dataset.copy()

# 训练数组
train_paragraph_data = list(train_dataset["段落"])
# 测试数组
test_paragraph_data = list(test_dataset["段落"])

print("训练段落的数组===>\n", list(train_paragraph_data))
print("测试段落的数组===>\n", list(test_paragraph_data))

# todo 未完待续
# 对目标文档分词，保存在all_doc_list
train_docs_split = []
test_docs_split = []
# 训练文档分词，保存在train_docs_list
for doc in train_paragraph_data:
    doc_list = [word for word in jieba.cut(doc)]
    train_docs_split.append(doc_list)
print("分词结果=>", train_docs_split)

# 测试文档分词，保存在test_docs_list
for doc in test_paragraph_data:
    doc_list = [word for word in jieba.cut(doc)]
    test_docs_split.append(doc_list)

# 制作语料库，词袋 bag-of-words
dict_lib = corpora.Dictionary(train_docs_split)
# print("打印一下语料库=>", dict_lib)
# print("词用数字进行编号关系=>", dict_lib.keys())
# print("编号与词的对应关系=>", dict_lib.token2id)

# # 使用doc2bow制作语料库
corpus = [dict_lib.doc2bow(doc) for doc in train_docs_split]
print("使用doc2bow制作训练语料库=>", corpus)

for doc in test_docs_split:
    doc_row = ''.join(str(i) for i in doc)
    docs_test_vec = dict_lib.doc2bow(doc)
    if len(docs_test_vec) > 0:
        print("使用doc2bow制作测试语料库=>", docs_test_vec)
        # 相似度分析，使用TF-IDF模型对语料库进行建模
        tfidf = models.TfidfModel(corpus)
        # 获取测试文档中，每个词的TF-IDF值
        # print("测试文档中TF-IDF值=>", tfidf[docs_test_vec])
        # 对每个目标文档，分析测试文档的相似度
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dict_lib.keys()))
        sim = index[tfidf[docs_test_vec]]
        # 根据相似度排序
        sortedArray = sorted(enumerate(sim), key=lambda item: -item[1]) or []
        index = str(sortedArray[0][0])
        print("\n当前段落可能是第：", row_json_map[index] + '段\n', doc_row)
    else:
        print("\n非法无效的词段==>\n", doc_row)

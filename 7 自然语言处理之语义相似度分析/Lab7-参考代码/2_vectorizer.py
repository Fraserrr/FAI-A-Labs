from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

# 1. 准备工作 (已在前面导入)

# 2. 输入三个以上句子
sentences_bow = [
    "我喜欢吃苹果，也喜欢香蕉。",
    "今天天气真好，适合出去玩。",
    "小明喜欢在公园里跑步，小红喜欢在家里看书。",
    "苹果是一种常见的水果，营养丰富。"
]


# 3. 分词
def jieba_cut(sentence):
    return " ".join(jieba.cut(sentence))


sentences_bow_cut = [jieba_cut(s) for s in sentences_bow]
print("分词结果:")
for s_cut in sentences_bow_cut:
    print(s_cut)
print("-" * 30)

# 4. 构建词袋模型
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(sentences_bow_cut)
# 打印词汇表，即模型学到的所有词语
# print("词汇表 (Vocabulary):")
# print(vectorizer.get_feature_names_out())
# 打印每个句子的词袋向量 (稀疏矩阵表示)
# print("词袋向量 (Bag-of-words Vectors):")
# print(X_bow.toarray())
# print("-" * 30)

# 5. 计算余弦相似度
cosine_sim_bow = cosine_similarity(X_bow)

print("句子间余弦相似度矩阵:")
print(cosine_sim_bow)

# 更友好地输出两两之间的相似度
for i in range(len(sentences_bow)):
    for j in range(i + 1, len(sentences_bow)):
        print(f"句子'{sentences_bow[i]}' 与 句子'{sentences_bow[j]}' 的余弦相似度为: {cosine_sim_bow[i, j]:.4f}")

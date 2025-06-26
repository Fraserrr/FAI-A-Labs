import jieba
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # 忽略gensim相关的弃用警告

# 设置中文字体，确保热图中的中文能正确显示 (根据系统和环境可能需要调整)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常见中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备工作 (已在前面导入)

# 2. 输入的句子列表 [cite: 3]
sentences_w2v = [
    "我喜欢在清晨喝一杯热茶，享受安静的时光。",
    "今天是我生日，我准备和家人一起庆祝。",
    "科技改变了我们的生活，尤其是人工智能的发展。",
    "早晨的阳光洒在大地上，给一切带来了温暖。",
    "我喜欢在每天下午来一杯咖啡，它能让我更加集中精神。",
    "每当下雨时，我喜欢在窗前静静地看着雨滴落下。",
    "昨晚我和朋友一起去看了一场电影，感觉很放松。",
    "电脑和手机已经成为现代社会不可或缺的一部分。",
    "他每天都在健身房锻炼，已经养成了好习惯。",
    "今天是我的生日，但我与朋友约好了一起在海底捞庆祝。",
    "我喜欢读科幻小说，因为它们带我进入了另一个未知的世界。"
]


# 3. 分词
def cut_sentences(sentences):
    return [list(jieba.cut(sentence)) for sentence in sentences]


tokenized_sentences = cut_sentences(sentences_w2v)
print("分词后的结果:")
for i, sentence_tokens in enumerate(tokenized_sentences):
    print(f"句子{i + 1}: {' '.join(sentence_tokens)}")
print("-" * 30)

# 4. 模型训练 [cite: 4]
# sg=0 表示 CBOW, sg=1 表示 Skip-gram
# vector_size 替代旧版的 size
model_w2v = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0, workers=4)  # [cite: 4]


# print("Word2Vec模型训练完成。")
# print(f"词汇表大小: {len(model_w2v.wv.key_to_index)}")
# print("-" * 30)

# 5. 计算句子向量 [cite: 5]
def sentence_vector(tokens):
    vectors = [model_w2v.wv[word] for word in tokens if word in model_w2v.wv]  # [cite: 5]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model_w2v.vector_size)  # [cite: 5]


sentence_vectors = [sentence_vector(sentence_tokens) for sentence_tokens in tokenized_sentences]  # [cite: 5]
# print("句子向量计算完成。")
# for i, vec in enumerate(sentence_vectors):
#     print(f"句子{i+1}的向量 (前5个维度): {vec[:5]}...")
# print("-" * 30)

# 6. 计算相似度
cosine_sim_w2v = cosine_similarity(sentence_vectors)
print("句子余弦相似度矩阵:")
print(np.round(cosine_sim_w2v, decimals=3))  # 打印时保留3位小数，方便查看
print("-" * 30)

# 7. 可视化结果
plt.figure(figsize=(12, 10))  # 调整图像大小以容纳更多标签
sns.heatmap(cosine_sim_w2v,
            annot=True,
            cmap="coolwarm",
            xticklabels=[f"句子{i + 1}" for i in range(len(sentences_w2v))],
            yticklabels=[f"句子{i + 1}" for i in range(len(sentences_w2v))],
            fmt=".2f")
plt.title("句子余弦相似度热图 (Word2Vec)")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，防止重叠
plt.yticks(rotation=0)
plt.tight_layout()  # 自动调整布局
plt.savefig(f"句子余弦相似度热图 (Word2Vec)")
# plt.show()

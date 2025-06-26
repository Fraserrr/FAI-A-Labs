import numpy as np

# 从表格中提取苹果和大米的数据 [cite: 1]
# 特征顺序: 水果, 公司, 手机, 粮食
R_pingguo = np.array([0.96, 0.77, 0.85, 0.15])  # 苹果的特征向量
R_dami = np.array([0.18, 0.22, 0.05, 0.93])    # 大米的特征向量

# 计算余弦相似度
cosine_similarity_value = np.dot(R_pingguo, R_dami) / (np.linalg.norm(R_pingguo) * np.linalg.norm(R_dami))

# 计算广义Jaccard相似度
dot_product = np.dot(R_pingguo, R_dami)
norm_pingguo_sq = np.linalg.norm(R_pingguo)**2
norm_dami_sq = np.linalg.norm(R_dami)**2
jaccard_similarity_value = dot_product / (norm_pingguo_sq + norm_dami_sq - dot_product)

print(f"苹果和大米的特征向量:")
print(f"R_苹果 = {R_pingguo}")
print(f"R_大米 = {R_dami}")
print(f"苹果和大米的余弦相似度: {cosine_similarity_value:.4f}")
print(f"苹果和大米的广义Jaccard相似度: {jaccard_similarity_value:.4f}")
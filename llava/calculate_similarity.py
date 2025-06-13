import numpy as np
from tqdm import tqdm

# path1="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/layer-1_output_layer.npy"
# path2="/mnt/petrelfs/chenjingzhou/cjz/Minerulz/layer-1_output_layer.npy"

# path1="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/layer-1_output_int.npy"
# path2="/mnt/petrelfs/chenjingzhou/cjz/Minerulz/layer-1_output_int.npy"

# path1="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/layer-1_output_head.npy"
# path2="/mnt/petrelfs/chenjingzhou/cjz/Minerulz/layer-1_output_head.npy"

path1="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/layer-1_output_all.npy"
path2="/mnt/petrelfs/chenjingzhou/cjz/Minerulz/layer-1_output_all.npy"

layer0_output1 = np.load(path1)
layer0_output2 = np.load(path2)

# 已知layer0_output形状为[1, token_num, dim]
# 逐token计算余弦相似度，然后统计相似度的平均值

print(f"Layer 0 output 1 shape: {layer0_output1.shape}")
print(f"Layer 0 output 2 shape: {layer0_output2.shape}")

def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度
    :param a: 向量a
    :param b: 向量b
    :return: 余弦相似度
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

similarities = []
for token1, token2 in zip(layer0_output1[0], layer0_output2[0]):
# for token1 in tqdm(layer0_output1[0]):
#     for token2 in layer0_output2[0]:
    similarity = cosine_similarity(token1, token2)
    similarities.append(similarity)

# 打印最大的5个相似度及其idx
# sorted_indices = np.argsort(similarities)[-100:]
# 改为打印最小的5个相似度及其idx
sorted_indices = np.argsort(similarities)[:100]
print("Top 5 cosine similarities and their indices:")
for idx in sorted_indices:
    print(f"Index: {idx}, Cosine Similarity: {similarities[idx]}")


avg_similarity = np.mean(similarities)
print(f"Average cosine similarity: {avg_similarity}")


# 计算每个位置的向量模长的比值，打印所有比值
for i in range(len(layer0_output1[0])):
    norm1 = np.linalg.norm(layer0_output1[0][i])
    norm2 = np.linalg.norm(layer0_output2[0][i])
    if norm1 and norm2:
        ratio = norm1 / norm2
    else:
        ratio = 0.0
    print(f"Position {i}: Norm1: {norm1}, Norm2: {norm2}, Ratio: {ratio}")
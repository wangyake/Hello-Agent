import numpy as np

embeddings = {
    'king':np.array([0.9, 0.8]),
    'queen':np.array([0.9, 0.2]),
    'man':np.array([0.7, 0.9]),
    'woman':np.array([0.7, 0.3]),
}

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

result_vec = embeddings['king'] - embeddings['man'] + embeddings['woman']

sim = cosine_similarity(result_vec, embeddings['queen'])
print(f'king - man + woman = {sim:.4f}')

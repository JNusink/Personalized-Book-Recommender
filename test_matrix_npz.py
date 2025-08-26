from scipy.sparse import load_npz
import numpy as np

matrix = load_npz('user_item_matrix.npz')
indices = np.load('user_item_indices.npy', allow_pickle=True)
columns = np.load('user_item_columns.npy', allow_pickle=True)
print("Matrix shape:", matrix.shape)
print("Indices sample:", indices[:10])
print("Columns sample:", columns[:10])

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec
import networkx as nx

def make_graph(time_matrix, machine_matrix):
    G = nx.Graph()
    num_jobs, num_machines = time_matrix.shape
    for i in range(num_jobs):
        for j in range(num_machines):
            G.add_edge(f'job_{i}', f'machine_{machine_matrix[i, j]}', weight=time_matrix[i, j])
    return G

def generate_embeddings(x, dimensions=20, walk_length=30, num_walks=200, window=10, min_count=1, batch_words=4, p=1, q=1, workers=4):
    embeddings = []
    for i in range(x.shape[0]):
        time_matrix = x[i, :, :, 0]
        machine_matrix = x[i, :, :, 1]
        G = make_graph(time_matrix, machine_matrix)
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        emb = np.array([model.wv[str(node)] for node in G.nodes()])
        embeddings.append(emb)
    return np.array(embeddings)

if __name__ == '__main__':
    # 데이터 로드
    data = np.load('/media/NAS/USERS/moonbo/jssp/jssp_supervision_data.npz')
    x = data['x']  # (50000, 100, 20, 2)
    y = data['y']  # (50000, )

    # 데이터를 torch tensor로 변환
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 데이터셋을 8:1:1 비율로 학습, 검증, 테스트 세트로 분할
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    # 학습, 검증, 테스트 데이터에 대해 임베딩 생성
    train_embeddings = generate_embeddings(train_x)
    val_embeddings = generate_embeddings(val_x)
    test_embeddings = generate_embeddings(test_x)

    print("Train embeddings shape:", train_embeddings.shape)
    print("Validation embeddings shape:", val_embeddings.shape)
    print("Test embeddings shape:", test_embeddings.shape)

    # TensorDataset으로 변경
    train_dataset = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))

    # 데이터 로더 생성
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Encoder
    model = UTransformerEncoder(d_model=d_model, nhead=8, num_layers=[4, 4, 4], dim_feedforward=2048, dropout=0.1, patch_sizes=[20, 10, 10], pixel_sizes=[4, 4, 4])

    # Decoder
    # actor-critic module 사용해야할 듯?


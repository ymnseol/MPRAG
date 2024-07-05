import faiss
import numpy as np
import torch
import torch.nn.functional as F


def random_walk_gpu(adj_matrix, start_nodes, walk_length):
    walks = []
    for start_node in start_nodes:
        walk = [start_node]

        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = torch.where(adj_matrix[current_node] > 0)[0]

            if len(neighbors) == 0:
                break

            weights = adj_matrix[current_node, neighbors]
            probabilities = weights / weights.sum()
            next_node = neighbors[torch.multinomial(probabilities, 1).item()]
            walk.append(next_node.item())
        walks.append(walk)

    return walks


def make_sim_graph(selected_embeddings, top_k=200, threshold=0.75, device='cpu'):
    selected_embeddings = selected_embeddings.to(device)
    adj_matrix = torch.zeros((top_k, top_k), device=device)

    for i in range(top_k):
        similarities = F.cosine_similarity(selected_embeddings[i].unsqueeze(0), selected_embeddings, dim=1)
        mask = similarities > threshold
        adj_matrix[i, mask] = similarities[mask]

    return adj_matrix


def sort_sequence_by_similarity(sequence, selected_embeddings, top_k=200):
    similarity_matrix = F.cosine_similarity(selected_embeddings.unsqueeze(1), selected_embeddings.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix.cpu().numpy()

    sorted_sequence = [sequence[0]]
    sequence_set = set(sequence)
    sequence_set.remove(sequence[0])

    current_node = sequence[0]
    while sequence_set:
        next_node = max(sequence_set, key=lambda x: float(similarity_matrix[current_node, x]))
        sorted_sequence.append(next_node)
        sequence_set.remove(next_node)
        current_node = next_node

        if len(sorted_sequence) >= top_k:
            break

    return sorted_sequence


def find_passage(index, query_embeddings, dim=768, top_k=500, threshold=0.75, walk_length=199, num_walks=50):
    device = query_embeddings.device

    if device != 'cpu':
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    total_passage_embeddings = []
    batch_size = query_embeddings.shape[0]

    query_embeddings_for_faiss = query_embeddings.detach().cpu().numpy()
    query_embeddings_for_faiss = np.ascontiguousarray(query_embeddings_for_faiss)

    D, I = index.search(query_embeddings_for_faiss, top_k)

    for query_idx in range(0, batch_size):
        selected_indices = I[query_idx]
        selected_embeddings = np.array([index.reconstruct(int(idx)) for idx in selected_indices])
        selected_embeddings = torch.tensor(selected_embeddings, dtype=torch.float32).to(device)

        adj_matrix = make_sim_graph(selected_embeddings, top_k=top_k, threshold=threshold, device=device)

        central_node = 0

        walks = random_walk_gpu(adj_matrix, [central_node] * num_walks, walk_length)

        unique_sequence = []
        seen = set()
        for walk in walks:
            for node in walk:
                if node not in seen:
                    unique_sequence.append(node)
                    seen.add(node)

        sorted_sequence = sort_sequence_by_similarity(unique_sequence, selected_embeddings, top_k=100)
        sorted_embeddings = selected_embeddings[sorted_sequence]
        sorted_embeddings = sorted_embeddings.view(1, -1, dim)

        total_passage_embeddings.append(sorted_embeddings)

    total_passage_embeddings = torch.cat(total_passage_embeddings, dim=0)
    query_embeddings = query_embeddings.reshape(-1, 1, dim)

    token_out = torch.cat([query_embeddings, total_passage_embeddings], dim=1)

    return token_out

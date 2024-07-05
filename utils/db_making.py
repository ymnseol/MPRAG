import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
from datasets import load_dataset


def get_dataset(data_name_1, data_name_2):
    dataset_1 = load_dataset(data_name_1)
    dataset_2 = load_dataset(data_name_2, 'v2.1')

    marco_passages = []

    for i in range(len(dataset_2['train'])):
        marco_passages.append(' '.join(dataset_2['train'][i]['passages']['passage_text']))

    for i in range(len(dataset_2['validation'])):
        marco_passages.append(' '.join(dataset_2['validation'][i]['passages']['passage_text']))

    for i in range(len(dataset_2['test'])):
        marco_passages.append(' '.join(dataset_2['test'][i]['passages']['passage_text']))

    squad_passages = []

    for i in range(len(dataset_1['train'])):
        squad_passages.append(dataset_1['train'][i]['context'])

    for i in range(len(dataset_1['validation'])):
        squad_passages.append(dataset_1['validation'][i]['context'])

    squad_passages = list(set(squad_passages))
    marco_passages = list(set(marco_passages))

    return squad_passages, marco_passages


def get_embeddings(data_loader, model, device, tokenizer):
    model.eval()
    all_embeddings = []

    with tqdm(total=len(data_loader)) as pbar:
        for batch in data_loader:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            cls_embeddings = outputs.pooler_output
            all_embeddings.append(cls_embeddings.cpu().numpy())

            pbar.update(1)

    return np.vstack(all_embeddings)


if __name__ == '__main__':
    data_name_1 = 'squad_v2'
    data_name_2 = 'ms_marco'

    squad_passages, marco_passages = get_dataset(data_name_1, data_name_2)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    squad_loader = DataLoader(squad_passages, batch_size=128, shuffle=False)
    marco_loader = DataLoader(marco_passages, batch_size=128, shuffle=False)

    squad_embeddings = get_embeddings(squad_loader, model, device, tokenizer)
    marco_embeddings = get_embeddings(marco_loader, model, device, tokenizer)

    dimension = squad_embeddings.shape[1]

    squad_index = faiss.IndexHNSWFlat(dimension, 32)
    squad_index.add(squad_embeddings)
    faiss.write_index(squad_index, 'squad.index')

    marco_index = faiss.IndexHNSWFlat(dimension, 32)
    marco_index.add(marco_embeddings)
    faiss.write_index(marco_index, 'marco.index')

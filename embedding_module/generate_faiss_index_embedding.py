from transformers import BertModel, BertTokenizer
import torch
import faiss
import numpy as np
import csv
from tqdm import tqdm
import time

class FaissEmbeddingGeneration :
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_cls_embedding(self, text) :
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # cls token
        
        return cls_embedding
    
    def generate_embeddings(self, passages, batch_size = 32) :
        embeddings = []
        total_batches = len(passages) // batch_size + (1 if len(passages) % batch_size != 0 else 0)
        start_time = time.time()

        for i in tqdm(range(0, len(passages), batch_size), desc="Processing batches", total=total_batches):
            batch = passages[i:i + batch_size]
            batch_embeddings = self.get_cls_embedding(batch)
            embeddings.append(batch_embeddings)
        
        embeddings = np.concatenate(embeddings, axis=0)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Embedding generation took {elapsed_time:.2f} seconds.")

        return embeddings
    
    def save_faiss_index(self, embeddings, file_name) :
        d = embeddings.shape[1]  # 768

        index = faiss.IndexHNSWFlat(d, 32)
        index.add(embeddings)

        faiss.write_index(index, f'{file_name}.bin')

        # tsv file
        with open(f'{file_name}.tsv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for idx, embedding in enumerate(embeddings):
                writer.writerow([idx] + embedding.tolist())
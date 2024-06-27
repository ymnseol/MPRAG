from generate_faiss_index_embedding import FaissEmbeddingGeneration
from datasets import load_dataset

batch_size = 32
squad_v2 = load_dataset('squad_v2')
passages = [item['context'] for item in squad_v2['train']]
file_name = 'faiss_index_with_squadv2_embeddings'

generator = FaissEmbeddingGeneration()
embeddings = generator.generate_embeddings(passages, batch_size)
generator.save_faiss_index(embeddings, file_name)
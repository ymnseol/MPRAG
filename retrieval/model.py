import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import data as dt
import index as id
import util
import pickle
from torch.utils.data import DataLoader


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    
    with open(embedding_files, 'rb') as fin:
        ids, embeddings = pickle.load(fin)

    allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
    allembeddings1 = allembeddings
    allids.extend(ids)
    while allembeddings.shape[0] > indexing_batch_size:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
    return allembeddings1

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection

class Retriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()
        self.index = index = id.Indexer(self.model.config.indexing_dimension) 
        self.embeddings = index_encoded_data(self.index,'/home/tako/june/reproduce/retrieval/data/wikipedia_embeddings',50000)

    
    
    
    def forward(self,
                question_ids,
                question_mask,
                gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        question_output=question_output.cpu().numpy()
        top_ids_and_scores = self.index.search_knn(question_output, 100) 
        for i in range(question_output.shape[0]):
            ids = [int(x) - 1 for x in top_ids_and_scores[i][0]]
            passage_output = self.embeddings[ids] 
        
        

        return question_output, passage_output

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
    

if __name__ == '__main__':
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = dt.load_data('/home/tako/june/reproduce/retrieval/data/test.jsonl')
    model_class = Retriever
    model = model_class.from_pretrained('/home/tako/june/reproduce/pretrained_model/nq_retriever')
    model.cuda()
    model.eval()
    
    batch_size = 1
    dataset = dt.Dataset(data)
    collator = dt.Collator(40, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, _, _, question_ids, question_mask) = batch
            output_q, output_p = model.forward(
                question_ids.cuda().view(-1, question_ids.size(-1)), 
                question_mask.cuda().view(-1, question_ids.size(-1)), 
            )
            from IPython import embed ; embed()
    
    


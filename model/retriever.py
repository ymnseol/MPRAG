import torch
import torch.nn as nn
from transformers import BertModel

from utils.search_passages import find_passage


class Retriever(nn.Module):
    def __init__(self,
                 index,
                 model_name='bert-base-uncased',
                 top_k=500,
                 threshold=0.75,
                 walk_length=199,
                 num_walks=50):
        super(Retriever, self).__init__()

        self.index = index
        self.top_k = top_k
        self.threshold = threshold
        self.walk_length = walk_length
        self.num_walks = num_walks

        self.encoder = BertModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        with torch.no_grad():
            out = self.encoder(kwargs['input_ids'])

        sentence_embeds = out.last_hidden_state
        out = out.pooler_output

        query_with_passage = find_passage(self.index, out, top_k=self.top_k, threshold=self.threshold,
                                          walk_length=self.walk_length, num_walks=self.num_walks)

        return sentence_embeds, query_with_passage

    def update(self, source_encoder, momentum=0.99):
        source_params = {name: param for name, param in source_encoder.named_parameters()}

        for name, target_param in self.named_parameters():
            if name in source_params:
                source_param = source_params[name]
                if target_param.data.shape == source_param.data.shape:
                    target_param.data = momentum * target_param.data + (1 - momentum) * source_param.data

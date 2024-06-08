import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class Reranker(nn.Module):
    def __init__(
        self,
        model_path: str,
        reranker_config,
    ):
        super().__init__()

        ### Config ###

        self.config = reranker_config

        ### Reranker ###

        # Cross-encoder: retriever로부터 (query + document) embedding을 받아 TODO
        self.model = AutoModel.from_pretrained(model_path)

        self.dropout = nn.Dropout(
            p=self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

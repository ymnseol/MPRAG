import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModel, AutoConfig
from transformers.utils import logging
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_outputs import TokenClassifierOutput


logger = logging.get_logger(__name__)


class Classifier(nn.Module):
    def __init__(
        self,
        reranker_config,
        encoder_model_path,
    ):
        super().__init__()

        ### Config ###

        self.reranker_config = reranker_config
        self.encoder_config = AutoConfig.from_pretrained(encoder_model_path)

        ### Input Layer ###

        # Embedding: reranker logit을 BERT Encoder input으로 변환
        self.embedding = nn.Linear(
            in_features=self.reranker_config.hidden_size,  # Reranker(BERT-base): 768
            out_features=self.encoder_config.hidden_size,  # BERT-base: 768
            bias=True,
        )  # -> (batch size, # (query + documents), hidden_size)

        # Positional encoding
        self.positional_encoding = nn.Embedding(
            num_embeddings=self.encoder_config.max_position_embeddings,  # BERT-base: 512
            embedding_dim=self.encoder_config.hidden_size,  # BERT-base: 768
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=self.encoder_config.hidden_size,  # BERT-base: 768
            eps=self.encoder_config.layer_norm_eps,  # BERT-base: 1e-12
        )
        self.dropout = nn.Dropout(
            p=(
                self.encoder_config.classifier_dropout
                if self.encoder_config.classifier_dropout
                else self.encoder_config.hidden_dropout_prob
            ),  # BERT-base: null (classifier_dropout), 0.1(hidden_dropout_prob)
        )  # -> (batch size, # (query + documents), hidden_size)

        ### Encoder Layer ###

        # Transformer encoder
        self.encoder = AutoModel.from_pretrained(
            encoder_model_path
        ).encoder  # -> last_hidden_states: (batch size, # (query + documents), hidden_size)

        # Token classifier: token(=document)별 query와의 연관도 판단
        self.classifier = nn.Linear(
            in_features=self.encoder_config.hidden_size,
            out_features=self.encoder_config.num_labels,
        )

        ### Loss ###
        # TODO: end-to-end 학습 시 수정 필요
        self.cross_entropy = nn.CrossEntropyLoss(
            reduction="mean",
        )

        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(
                end=self.encoder_config.max_position_embeddings,  # BERT-base: 512
            ).expand((1, -1)),
        )

    def forward(
        self,
        reranker_logits: torch.Tensor,  # (batch size, # (query + documents), reranker's hidden size)
        attetion_mask: torch.Tensor = None,  # (batch size, batch 내 가장 긴 # (query + documents)). 일반적으로 batch는 같은 top-k를 공유하기 때문에 별도의 attention mask를 필요로 하지 않음.
        labels=None,
    ) -> torch.Tensor:
        batch_size, num_tokens, reranker_hidden_size = (
            reranker_logits.size()
        )  # (batch size, # (query + documents), reranker hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward: reranker logits' size: {reranker_logits.size()}"
        )

        ### Input Layer ###

        # Reranker output을 embedding으로 변환
        input_embeddings = self.embedding(
            reranker_logits
        )  # -> (batch size, # (query + documents), hidden size
        logger.debug(
            f"DEBUG: model > HLATR > forward > Input Layer: input_embeddings' size: {input_embeddings.size()}"
        )

        # 변환된 embedding에 positional encoding 추가
        position_ids = self.position_ids[
            :, :num_tokens
        ]  # -> (batch size, # (query + documents))
        logger.debug(
            f"DEBUG: model > HLATR > forward > Input Layer: positional ids' size: {position_ids.size()}"
        )
        positional_embeddings = self.positional_encoding(
            position_ids
        )  # -> (batch size, # (query + documents), hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward > Input Layer: positional_embeddings' size: {positional_embeddings.size()}"
        )
        input_embeddings = (
            input_embeddings + positional_embeddings
        )  # -> (batch size, # (query + documents), hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward > Input Layer: input_embeddings' size AFTER POSITIONAL ENCODING: {input_embeddings.size()}"
        )

        input_embeddings = self.layer_norm(
            input_embeddings
        )  # -> (batch size, # (query + documents), hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward: input_embeddings' size AFTER LAYERNORM: {input_embeddings.size()}"
        )
        input_embeddings = self.dropout(
            input_embeddings
        )  # -> (batch size, # (query + documents), hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward: input_embeddings' size AFTER DROPOUT: {input_embeddings.size()}"
        )

        ### Encoder Layer ###

        # Attention mask 생성
        device = input_embeddings.device
        output = self.encoder(
            hidden_states=input_embeddings,  # (batch size, # (query + documents), hidden size)
        )
        encoder_hidden_states = output[
            "last_hidden_state"
        ]  # (batch size, # (query + documents), hidden size)
        logger.debug(
            f"DEBUG: model > HLATR > forward > Encoder Layer: encoder_hidden_states: {encoder_hidden_states.size()}"
        )

        # BERT Encoder에서 나온 hidden state를 logit으로 변환
        logits = self.classifier(encoder_hidden_states).squeeze(
            -1
        )  # -> (batch size, sequence length, num_labels(default: 2))
        logger.debug(
            f"DEBUG: model > HLATR > forward > Encoder Layer: logits: {logits.size()}"
        )

        # Loss 계산
        # TODO: end-to-end 학습 시 수정 필요
        if labels is not None:
            labels = labels.double()
            loss = self.cross_entropy(logits, labels)
        else:
            loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_hidden_states,
            attentions=output.attentions,
        )

    def save_pretrained(
        self,
        save_directory: str,
        save_config: bool = True,
        state_dict: dict = None,
        save_function: callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        pass  # TODO: 저장 기능 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

from model.retriever import Retriever


class MPRAG(nn.Module):
    def __init__(self, model_name_1='bert-base-uncased', classifier_path=None, index=None, top_k=500, threshold=0.75,
                 walk_length=199, num_walks=50):
        super(MPRAG, self).__init__()

        self.retriever = Retriever(model_name=model_name_1, index=index, top_k=top_k, threshold=threshold,
                                   walk_length=walk_length, num_walks=num_walks)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path='facebook/bart-base')

        if classifier_path is None:
            self.classifier = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', num_labels=2)
        else:
            self.classifier = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=classifier_path, num_labels=2)

        hidden_size = self.classifier.config.hidden_size
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, **kwargs):
        input_sequence, out = self.retriever(**kwargs)
        input_embeds = self.projection(out)

        out = self.classifier(inputs_embeds=input_embeds)

        mask = F.softmax(out.logits, dim=-1).argmax(dim=-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, input_embeds.size(-1))
        out = input_embeds * (mask == 1).float()

        gen_input_embeds = torch.cat([input_sequence, out[:, 1:]], dim=1)

        existing_attention_mask = kwargs['attention_mask']
        new_attention_mask = torch.ones(gen_input_embeds.shape[0], gen_input_embeds.shape[1], dtype=torch.long)
        new_attention_mask[:, :existing_attention_mask.shape[1]] = existing_attention_mask

        out = self.generator(inputs_embeds=gen_input_embeds,
                             attention_mask=new_attention_mask.to(gen_input_embeds.device),
                             labels=kwargs['labels'],
                             )

        return out

    def retriever_update(self, momentum=0.99):
        self.retriever.update(source_encoder=self.classifier, momentum=momentum)


if __name__ == '__main__':
    import faiss

    index = faiss.read_index('../squad.index')
    model = MPRAG(index=index, top_k=250, threshold=0.1, walk_length=199, num_walks=50, classifier_path='../classifier_weight').cuda()

    example_data = {
        'input_ids': torch.randint(0, 1000, (2, 512)),
        'attention_mask': torch.randint(0, 2, (2, 512)),
        'token_type_ids': torch.randint(0, 2, (2, 512)),
        'labels': torch.randint(0, 2, (2, 512)),
    }

    data = {key: value.cuda() for key, value in example_data.items()}

    out = model(**data)
    print(out)

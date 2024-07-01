import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_metric, load_from_disk
from transformers import (
    BertConfig,
    TrainingArguments,
    Trainer,
)
from args import get_args
from modeling_classifier import Classifier


def compute_metrics(output):
    logits, labels = output
    predictions = torch.argmax(logits, dim=-1).view(-1)
    references = labels.view(-1)

    return {
        "precision": precision.compute(predictions=predictions, references=references, zero_division=0),
        "recall": recall.compute(predictions=predictions, references=references, zero_division=0),
        "f1": f1.compute(predictions=predictions, references=references, zero_division=0),
    }


class ClassifierDataset(Dataset):
    def __init__(self, data_path, passage_data_path):
        self.pairs = load_from_disk(data_path)
        self.passages = pd.read_csv(passage_data_path, sep="\t")

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        question_embed = torch.tensor(pair["question_embedding"]).unsqueeze(0)
        passage_embeds = torch.tensor(self.passages[self.passages.index.isin(pair["passage_ids"])].values)
        inputs_embeds = torch.cat((question_embed, passage_embeds), dim=0).type(torch.float32)
        if self.pairs.features.get("labels"):
            item = {
                "inputs_embeds": inputs_embeds,
                "labels": [1] + pair["labels"],
            }
        else:
            item = {"inputs_embeds": inputs_embeds}
        return item

    def __len__(self):
        return len(self.pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity

    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")

    model_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
    model_config.label_smoothing = args.label_smoothing
    model = Classifier.from_pretrained("google-bert/bert-base-uncased", config=model_config)

    train_dataset = ClassifierDataset(args.train_data_path, args.passage_data_path, args.top_k)
    eval_dataset = ClassifierDataset(args.eval_data_path, args.passage_data_path, args.top_k)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=args.dataloader_drop_last,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

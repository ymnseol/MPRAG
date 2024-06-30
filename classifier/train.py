import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_metric
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
    def __init__(self, data_path, passage_data_path, top_k=10):
        self.top_k = top_k
        self.pairs = pd.read_csv(data_path, sep="\t", index_col=0)
        self.passages = pd.read_csv(passage_data_path, sep="\t", index_col=0)
        self.data = self.construct()

    def __getitem__(self, idx):
        if self.data["labels"]:
            item = {
                "inputs_embeds": self.data["inputs_embeds"][idx],
                "labels": self.data["labels"][idx],
            }
        else:
            item = {"inputs_embeds": self.data["inputs_embeds"][idx]}
        return item

    def __len__(self):
        return len(self.data["inputs_embeds"])

    def construct(self):
        # TODO: preprocess로 따로 빼기
        data = {"inputs_embeds": [], "labels": []}
        for i in range(0, len(self.pairs), self.top_k):
            pairs = self.pairs.iloc[i : i + self.top_k]
            passage_ids = pairs["passage_id"].tolist()
            sequences = self.passages.iloc[passage_ids][[f"embedding_{i}" for i in range(768)]].values
            data["inputs_embeds"].append(torch.tensor(sequences).type(torch.float32))
            if "label" in pairs.columns:
                labels = pairs["label"].tolist()
                data["labels"].append(torch.tensor(labels).type(torch.long))
        return data


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

    train_dataset = ClassifierDataset(args.data_path) # TODO
    eval_dataset = ClassifierDataset(args.data_path) # TODO

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

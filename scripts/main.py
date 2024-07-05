import os
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from model.mprag import MPRAG
from scripts.train import Trainer
from utils.data_preprocessing import preprocess_data
os.environ["WANDB_PROJECT"] = "MPRAG"
os.environ["WANDB_LOG_MODEL"] = "logs"


def main(args):
    lr = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    data_name = args.data_name
    sch = args.scheduler
    opt = args.optimizer
    gamma = args.gamma
    step_size = args.step_size
    pretrained = args.pretrained
    idx_path = args.index_path

    wandb.init(project="MPRAG", name=f'MPRAG_{num_epochs}_{lr}_{opt}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if data_name == 'squad':
        dataset = load_dataset('squad_v2')
    else:
        dataset = load_dataset("ms_marco", 'v2.1')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    index = faiss.read_index(idx_path)
    model = MPRAG(index=index, top_k=250, threshold=0.3, walk_length=199, num_walks=50, classifier_path=pretrained).to(device)

    dataset = dataset.map(preprocess_data,
                          batched=True,
                          remove_columns=dataset['train'].column_names,
                          fn_kwargs={'tokenizer': tokenizer},
                          load_from_cache_file=False
                          )

    train_dataset = dataset['train']

    num_train = int(len(train_dataset) * 0.9)
    num_val = len(train_dataset) - num_train

    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    test_dataset = dataset['validation']

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    trainer = Trainer(model=model,
                      optimizer=opt,
                      scheduler=sch,
                      lr=lr,
                      gamma=gamma,
                      device=device,
                      step_size=step_size,
                      )

    print("Training model")
    trainer.train(model, train_loader, val_loader, num_epochs, accumulation_step=4, num_patience=5)
    print("Testing model")
    trainer.test(model, test_loader)

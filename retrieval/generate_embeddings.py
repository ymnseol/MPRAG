# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

import model
import data
import util





def embed_passages(passages, model, tokenizer):
    batch_size = 2048
    collator = data.TextCollator(tokenizer, model.config.passage_maxlength)
    dataset = data.TextDataset(passages, title_prefix='title:', passage_prefix='context:')
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            embeddings = model.embed_text(
                text_ids=text_ids.cuda(), 
                text_mask=text_mask.cuda(), 
                apply_mask=model.config.apply_passage_mask,
                extract_cls=model.config.extract_cls,
            )
            embeddings = embeddings.cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)


    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = [x for idlist in allids for x in idlist]
    return allids, allembeddings





if __name__ == '__main__':
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    model_class = model.Retriever
    model = model_class.from_pretrained('/home/tako/june/reproduce/pretrained_model/nq_retriever')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('start')
    passages = util.load_passages('/home/tako/june/reproduce/retrieval/corpus/psgs_w100.tsv')

    allids, allembeddings = embed_passages(passages, model, tokenizer)

    output_path = Path('wikipedia_embeddings')
    save_file = output_path.parent / (output_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)
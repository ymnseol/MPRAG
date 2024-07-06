import os
import random
import inspect
from tqdm import tqdm
from collections import Counter
import re
import string

import torch

import numpy as np
import wandb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Check if the normalized predictions and ground truths match exactly."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """Compute F1 score between the normalized predictions and ground truths."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    if precision + recall == 0:
        return 0

    return (2 * precision * recall) / (precision + recall)


def calculate_metrics_batch(predictions, labels, tokenizer):
    em_scores = []
    f1_scores = []

    for pred_ids, label_ids in zip(predictions, labels):
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True)

        em = exact_match_score(pred_text, label_text)
        f1 = f1_score(pred_text, label_text)

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)

    return avg_em, avg_f1


class Trainer:
    def __init__(self, **kwargs):
        self.seed = kwargs.get('seed', 42)
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = kwargs.get('tokenizer', None)

        seed_everything(self.seed)

        model = kwargs.get('model', None)
        optimizer = kwargs.get('optimizer', None)
        scheduler = kwargs.get('scheduler', None)

        if optimizer is not None and model is not None:
            optimizer_class = getattr(torch.optim, optimizer)
            optimizer_params = inspect.signature(optimizer_class).parameters

            valid_kwargs = {k: v for k, v in kwargs.items() if k in optimizer_params}

            self.optimizer = optimizer_class(model.parameters(), **valid_kwargs)
        else:
            self.optimizer = None

        if scheduler is not None and self.optimizer is not None:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)
            scheduler_params = inspect.signature(scheduler_class).parameters

            valid_kwargs = {k: v for k, v in kwargs.items() if k in scheduler_params and k != 'optimizer'}

            self.scheduler = scheduler_class(self.optimizer, **valid_kwargs)
        else:
            self.scheduler = None

    def train(self, model, train_loader, val_loader, num_epochs, accumulation_step=1, num_patience=5):
        best_loss = float('inf')
        patience = 0

        for epoch in range(num_epochs):
            self._train_step(model, train_loader, accumulation_step)
            val_loss, _, _ = self._val_step(model, val_loader)

            os.makedirs('checkpoints', exist_ok=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')

                patience = 0
            else:
                patience += 1

            if patience > num_patience:
                break

    def _train_step(self, model, train_loader, accumulation_step):
        train_loss = 0
        preds = list()
        labels = list()

        with tqdm(train_loader, unit='b', ascii=True, ncols=150, desc=f'Train') as pbar:
            model.train()

            for i, data in enumerate(train_loader):
                data = {k: v.to(self.device) for k, v in data.items()}

                output = model(**data)
                loss = output.loss
                train_loss += loss.item()

                if accumulation_step > 1:
                    loss = loss / accumulation_step

                loss.backward()
                model.retriever_update()

                if (i + 1) % accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                preds += list(output.logits.argmax(-1))
                labels += list(data['labels'].detach().cpu().numpy().tolist())

                pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                pbar.update(1)

        em, f1 = calculate_metrics_batch(preds, labels, self.tokenizer)

        wandb.log({'train/loss': train_loss / len(train_loader),
                   'train/f1': f1,
                   'train/em': em
                   })

    def _val_step(self, model, val_loader):
        val_loss = 0

        preds = list()
        labels = list()

        with tqdm(val_loader, unit='b', ascii=True, ncols=150, desc=f'Validation') as pbar:
            model.eval()

            for i, data in enumerate(val_loader):
                data = {k: v.to(self.device) for k, v in data.items()}

                with torch.no_grad():
                    output = model(**data)
                    loss = output.loss
                    val_loss += loss.item()

                    preds += list(output.logits.argmax(-1))
                    labels += list(data['labels'].detach().cpu().numpy().tolist())

                    pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                    pbar.update(1)

        em, f1 = calculate_metrics_batch(preds, labels, self.tokenizer)

        wandb.log({'val/loss': val_loss / len(val_loader),
                   'val/f1': f1,
                   'val/em': em
                   })

        return val_loss / len(val_loader), f1, em

    def test(self, model, test_loader):
        model.load_state_dict(torch.load('checkpoints/best_model.pth'))
        test_loss, f1, em = self._val_step(model, test_loader)

        print(f'test loss: {test_loss:.4f}, f1: {f1}, em: {em}')

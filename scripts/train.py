import os
import random
import inspect
from tqdm import tqdm

import torch
from evaluate import load

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


class Trainer:
    def __init__(self, **kwargs):
        self.seed = kwargs.get('seed', 42)
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.m1 = load('bleu')
        self.m2 = load('meteor')
        self.m3 = load('rouge')

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
            val_loss, _, _, _ = self._val_step(model, val_loader)

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

                if (i + 1) % accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                preds += list(output.logits.argmax(-1))
                labels += list(data['labels'].detach().cpu().numpy())

                pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                pbar.update(1)

        bleu = self.m1.compute(predictions=preds, references=labels)
        meteor = self.m2.compute(predictions=preds, references=labels)
        rouge = self.m3.compute(predictions=preds, references=labels)

        wandb.log({'train/loss': train_loss / len(train_loader),
                   'train/bleu': bleu,
                   'train/meteor': meteor,
                   'train/rouge': rouge
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
                    labels += list(data['labels'].detach().cpu().numpy())

                    pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                    pbar.update(1)

        bleu = self.m1.compute(predictions=preds, references=labels)
        meteor = self.m2.compute(predictions=preds, references=labels)
        rouge = self.m3.compute(predictions=preds, references=labels)

        wandb.log({'val/loss': val_loss / len(val_loader),
                   'val/bleu': bleu,
                   'val/meteor': meteor,
                   'val/rouge': rouge
                   })

        return val_loss / len(val_loader), bleu, meteor, rouge

    def test(self, model, test_loader):
        model.load_state_dict(torch.load('checkpoints/best_model.pth'))
        test_loss, bleu, meteor, rouge = self._val_step(model, test_loader)

        print(f'test loss: {test_loss:.4f}, bleu: {bleu}, meteor: {meteor}, rouge: {rouge}')

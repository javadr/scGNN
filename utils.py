import numpy as np
import copy
from typing import Tuple, Union, Optional
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from rich import print

import torch
import shutil
import time

from config import CFG

sec2time = lambda secs: f"{int(secs/60):02}:{int(secs%60):02}"

def train_val_test(data, ratio_train: float, ratio_val_to_test: float) -> Tuple:
    ratio_val = ratio_val_to_test / (1 + ratio_val_to_test)
    train, test = train_test_split(data, train_size=ratio_train)
    val, test = train_test_split(test, train_size=ratio_val)

    return train, val, test


def mask(data: np.matrix, masked_prob: float) -> Tuple:
    index_pair = np.where(data != 0)
    size = index_pair[0].size
    masking_idx = np.random.choice(size, int(size*masked_prob), replace=False)
    masked_data = copy.deepcopy(data)
    # to retrieve the position of the masked:
    masked_data[index_pair[0][masking_idx], index_pair[1][masking_idx]] = 0
    return masked_data, index_pair, masking_idx


class Visualize:
    def __init__(self, output: Union[np.ndarray, list, dict], target: Optional[np.ndarray] = None):
        self.output = output
        self.target = target

    def plot(self, **kwargs):
        fig = plt.figure()
        if isinstance(self.output, dict):
            for key, values in self.output.items():
                plt.plot(values, label=key)
            plt.legend()
        else:
            plt.plot(self.output)
        self._savefig(fig, **kwargs)

    def regplot(self, **kwargs) -> None:
        fig = plt.figure()
        sns.regplot(x=self.target, y=self.output, line_kws={
                    "color": "r", "alpha": 0.7, "lw": 5})
        self._savefig(fig, **kwargs)

    def _savefig(self, fig, **kwargs):
        plt.title(kwargs['title'])
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])
        fig.savefig(kwargs['savefigname'])


class Runner:
    def __init__(self, model, criterion, optimizer, **kwargs):
        self.model = model
        self.best_model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.x = kwargs['x']
        self.edge_index = kwargs['edge_index']
        self.gd = kwargs['gd']
        self.data = kwargs['data']

    def train_full(self, epochs: int, method: str) -> dict:
        train_time = AverageMeter()
        losses = AverageMeter()
        logs = defaultdict(list)
        best_val_loss = 1e4

        for epoch in range(0, epochs):
            begin = time.time()

            # switch to train mode
            self.model.train()
            self.model.zero_grad()
            train_loss = 0

            # feed into the model
            pred = self.model(self.x, self.edge_index)

            # compute output
            mask = self.gd.train_mask
            if method == 'full':
                output = pred[mask]
                target = self.data[mask]
            elif method == 'masked':  # 'masked'
                output = pred[mask][self.gd.index_mask[mask]]
                target = self.data[mask][self.gd.index_mask[mask]]
            else:  # 'zeros'
                output = pred[mask][self.gd.index_zeros[mask]]
                target = self.data[mask][self.gd.index_zeros[mask]]

            # calculate the gradient, update the weights
            loss = self.criterion(target, output)
            if epoch != 0:
                loss.backward()
                self.optimizer.step()
            train_loss = loss.item()
            losses.update(train_loss)

            full_loss, masked_loss, zeros_loss = self._triple_loss(
                self.gd.val_mask)

            loss = sum([full_loss, masked_loss, zeros_loss])/3
            if loss <= best_val_loss:
                best_val_loss = loss
                self.best_model = copy.deepcopy(self.model)
            for key, value in zip(('val_full', 'val_mask', 'val_zeros', 'train_loss'),
                                  (full_loss, masked_loss, zeros_loss, train_loss)):
                logs[key].append(value)

            # measure elapsed time
            train_time.update(time.time() - begin)

            if epoch <= 5 or epoch % 10 == 0:
                print(f'Epoch: [{epoch}]\t'
                      f'Training Loss (Avg) {losses.val:.4f} ({losses.avg:.4f}) / '
                      f'Time Elapsed[Avg] {sec2time(train_time.sum)}[{train_time.avg:.3f}s]\n'
                      f'Validation Loss (Full/Masked/Zeros) {full_loss:.4f}, {masked_loss:.4f}, {zeros_loss:.4f}'
                      )

        return logs

    @torch.inference_mode()
    def evaluate(self, method):
        eval_time = AverageMeter()
        losses = AverageMeter()

        begin = time.time()

        self.model.eval()
        pred = self.best_model(self.x, self.edge_index)
        mask = self.gd.test_mask
        if method == 'full':
            target = self.data[mask]
            output = pred[mask]
        elif method == 'masked':
            target = self.data[mask][self.gd.index_mask[mask]]
            output = pred[mask][self.gd.index_mask[mask]]
        elif method == 'zeros':
            target = self.data[mask][self.gd.index_zeros[mask]]
            output = pred[mask][self.gd.index_zeros[mask]]

        loss = self.criterion(output, target)

        eval_time.update(time.time()-begin)
        losses.update(loss.item())

        return output, target, loss.item()

    @torch.inference_mode()
    def predict(self):
        return self.evaluate(method='zeros')[:2]

    @torch.inference_mode()
    def _triple_loss(self, mask):
        self.model.eval()
        pred = self.model(self.x, self.edge_index)

        # Full
        target_full = self.data[mask]
        output_full = pred[mask]
        # Masked
        target_masked = self.data[mask][self.gd.index_mask[mask]]
        output_masked = pred[mask][self.gd.index_mask[mask]]
        # Zeros
        target_zeros = self.data[mask][self.gd.index_zeros[mask]]
        output_zeros = pred[mask][self.gd.index_zeros[mask]]

        loss_full = self.criterion(output_full, target_full)
        loss_masked = self.criterion(output_masked, target_masked)
        loss_zeros = self.criterion(output_zeros, target_zeros)

        return loss_full.item(), loss_masked.item(), loss_zeros.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

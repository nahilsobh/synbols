from typing import Optional, Callable

import numpy as np
import structlog
import torch
from baal.ensemble import EnsembleModelWrapper
from baal.utils.cuda_utils import to_cuda
from torch.utils.data import DataLoader

log = structlog.get_logger('CSGHWrapper')


class CSGHMCWrapper(EnsembleModelWrapper):
    def __init__(self, model, criterion, learning_epoch):
        super().__init__(model, criterion)
        self.weight_decay = 5e-4
        self.alpha = 0.9
        self.temperature = 1.0 / 50000
        self.lr_0 = 0.5
        self.T = learning_epoch
        self.M = 4
        self.discovery = self.T // self.M
        self.sampling = int(0.8 * self.discovery)

    def update_params(self, lr, epoch):
        for p in self.model.parameters():
            if not hasattr(p, 'buf'):
                p.buf = torch.zeros(p.size()).cuda()
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1 - self.alpha) * p.buf - lr * d_p
            if (epoch % self.discovery) + 1 > self.sampling:
                eps = torch.randn(p.size()).cuda()
                buf_new += (2.0 * lr * self.alpha * self.temperature) ** .5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def adjust_learning_rate(self, epoch, batch_idx, batch_total):
        rcounter = epoch * batch_total + batch_idx
        T = self.T * batch_idx
        cos_inner = np.pi * (rcounter % (T // self.M))
        cos_inner /= T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        return lr

    def train_on_dataset(self, dataset, optimizer, batch_size, epoch, use_cuda, workers=4,
                         regularizer: Optional[Callable] = None):
        self._weights.clear()
        self.train()
        history = []
        log.info("Starting training", epoch=epoch, dataset=len(dataset))
        for epoch_idx in range(epoch):
            self._reset_metrics('train')
            for idx, (data, target) in enumerate(
                DataLoader(dataset, batch_size, True, num_workers=workers)):
                _ = self.train_on_batch(epoch_idx, idx, data, target, optimizer, use_cuda,
                                        regularizer)
            history.append(self.metrics['train_loss'].value)
            if (epoch_idx % self.discovery) + 1 > self.sampling:
                self.add_checkpoint()

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info("Training complete", train_loss=self.metrics['train_loss'].value)
        return history

    def train_on_batch(self, epoch, batch_idx, data, target, optimizer, cuda=False,
                       regularizer: Optional[Callable] = None):
        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        lr = self.adjust_learning_rate(epoch, batch_idx)
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()
        self.update_params(lr, epoch)

        optimizer.step()
        self._update_metrics(output, target, loss, filter='train')
        return loss

from typing import Optional, Callable

import numpy as np
import structlog
import torch
from baal.ensemble import EnsembleModelWrapper
from baal.utils.cuda_utils import to_cuda
from torch.utils.data import DataLoader

log = structlog.get_logger('CSGHWrapper')


class CSGHMCWrapper(EnsembleModelWrapper):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)
        self.weight_decay = 5e-4
        self.alpha = 0.9
        self.num_cycles = 3
        self.lr_0 = 0.01
        self.T = None

    def update_params(self, lr, epoch, num_epoch):
        for p in self.model.parameters():
            if not hasattr(p, 'buf'):
                p.buf = torch.zeros(p.size()).cuda()
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1 - self.alpha) * p.buf - lr * d_p
            if (epoch / num_epoch) > 0.8:
                eps = torch.randn(p.size()).cuda()
                buf_new += (2.0 * lr * self.alpha * self.temperature) ** .5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def adjust_learning_rate(self, epoch, batch_idx, batch_total, epoch_total):
        rcounter = epoch * batch_total + batch_idx
        cos_inner = np.pi * (rcounter)
        cos_inner /= epoch_total * batch_total
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        return lr

    def train_on_dataset(self, dataset, optimizer, batch_size, epoch, use_cuda, workers=4,
                         regularizer: Optional[Callable] = None):
        self.clear_checkpoints()
        self.train()

        history = []
        log.info("Starting training", epoch=epoch, dataset=len(dataset))
        self.temperature = 1.0 / len(dataset)
        for cycle in range(self.num_cycles):
            for epoch_idx in range(epoch):
                self._reset_metrics('train')
                dl = DataLoader(dataset, batch_size, True, num_workers=workers)
                num_batch = len(dl)
                self.T = epoch * num_batch
                for idx, (data, target) in enumerate(dl):
                    _ = self.train_on_batch((epoch_idx, idx, num_batch, epoch), data, target, optimizer,
                                            use_cuda,
                                            regularizer)
                history.append(self.metrics['train_loss'].value)
                if (epoch - epoch_idx) <= 3:
                    self.add_checkpoint()

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info("Training complete", train_loss=self.metrics['train_loss'].value)
        return history

    def train_on_batch(self, lr_data, data, target, optimizer, cuda=False,
                       regularizer: Optional[Callable] = None):
        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        lr = self.adjust_learning_rate(*lr_data)
        self.model.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()
        self.update_params(lr, lr_data[0], lr_data[-1])

        self._update_metrics(output, target, loss, filter='train')
        return loss

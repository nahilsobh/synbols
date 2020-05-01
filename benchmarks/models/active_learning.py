import os
import warnings
from copy import deepcopy

import h5py
import numpy as np
import torch
from baal import ModelWrapper
from baal.active import ActiveLearningLoop, get_heuristic
from baal.active.heuristics import BALD, requireprobs, xlogy, BatchBALD
from baal.bayesian.dropout import patch_module
from baal.calibration import DirichletCalibrator
from baal.utils.metrics import ClassificationReport
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models

from datasets import get_dataset

pjoin = os.path.join

LOG_UNCERT = False


class BetaBALD(BALD):
    def __init__(self, beta, *args, **kwargs):
        self.beta = beta
        super().__init__(*args, **kwargs)

    @requireprobs
    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        assert predictions.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]

        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1),
                                     axis=-1)  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                      axis=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - self.beta * expected_entropy
        return bald_acq


class ActiveLearning(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.backbone = models.vgg16(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = torch.nn.Linear(num_ftrs, exp_dict["num_classes"])
        self.backbone = patch_module(self.backbone)
        self.initial_weights = deepcopy(self.backbone.state_dict())
        self.backbone.cuda()

        self.batch_size = exp_dict['batch_size']
        self.calibrate = exp_dict.get('calibrate', False)
        self.learning_epoch = exp_dict['learning_epoch']
        self.optimizer = torch.optim.SGD(self.backbone.parameters(),
                                         lr=exp_dict['lr'],
                                         weight_decay=5e-4,
                                         momentum=0.9,
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    verbose=True)
        self.criterion = CrossEntropyLoss()
        shuffle_prop = exp_dict.get('shuffle_prop', 0.0)
        if exp_dict['heuristic'] == 'betabald':
            max_sample = -1
            self.heuristic = BetaBALD(beta=exp_dict['beta'], shuffle_prop=shuffle_prop)
        elif exp_dict['heuristic'] == 'batch_bald':
            max_sample = 10000
            self.heuristic = BatchBALD(num_samples=exp_dict['query_size'],
                                       shuffle_prop=shuffle_prop)
        else:
            max_sample = -1
            self.heuristic = get_heuristic(exp_dict['heuristic'], shuffle_prop=shuffle_prop)
        self.wrapper = ModelWrapper(self.backbone, criterion=self.criterion)
        self.wrapper.add_metric('cls_report', lambda: ClassificationReport(exp_dict["num_classes"]))
        self.loop = ActiveLearningLoop(None, self.wrapper.predict_on_dataset,
                                       heuristic=self.heuristic,
                                       ndata_to_label=exp_dict['query_size'],
                                       batch_size=1,
                                       iterations=exp_dict['iterations'],
                                       use_cuda=True,
                                       max_sample=max_sample)

        self.calib_set = get_dataset('calib', exp_dict['dataset'])
        self.valid_set = get_dataset('val', exp_dict['dataset'])
        self.calibrator = DirichletCalibrator(self.wrapper, exp_dict["num_classes"],
                                              lr=0.001, reg_factor=exp_dict['reg_factor'],
                                              mu=exp_dict['mu'])

        self.active_dataset = None
        self.active_dataset_settings = None

    def train_on_loader(self, loader: DataLoader):
        self.wrapper.load_state_dict(self.initial_weights)
        if self.active_dataset is None:
            self.active_dataset = loader.dataset
            if self.active_dataset_settings is not None:
                self.active_dataset.load_state_dict(self.active_dataset_settings)
            self.loop.dataset = self.active_dataset
        self.criterion.train()
        self.wrapper.train_on_dataset(self.active_dataset, self.optimizer, self.batch_size,
                                      epoch=self.learning_epoch, use_cuda=True)

        metrics = self.wrapper.metrics
        return self._format_metrics(metrics, 'train')

    def val_on_loader(self, loader, savedir=None):
        val_data = loader.dataset
        self.loop.step()
        self.criterion.eval()
        self.wrapper.test_on_dataset(val_data, batch_size=self.batch_size, use_cuda=True)
        metrics = self.wrapper.metrics
        mets = self._format_metrics(metrics, 'test')
        mets.update({'num_samples': len(self.active_dataset)})
        return mets

    def on_train_end(self, savedir, epoch):
        h5_path = pjoin(savedir, 'ckpt.h5')
        labelled = self.active_dataset.state_dict()['labelled']
        with h5py.File(h5_path, 'a') as f:
            if f'epoch_{epoch}' not in f:
                g = f.create_group(f'epoch_{epoch}')
                g.create_dataset('labelled', data=labelled.astype(np.bool))

    def _format_metrics(self, metrics, step):
        mets = {k: v.value for k, v in metrics.items() if step in k}
        mets_unpacked = {}
        for k, v in mets.items():
            if isinstance(v, float):
                mets_unpacked[k] = v
            elif isinstance(v, np.ndarray):
                mets_unpacked[k] = v.mean()
            else:
                mets_unpacked.update({f"{k}_{ki}": np.mean(vi) for ki, vi in v.items()})
        return mets_unpacked

    def get_state_dict(self):
        state = {}
        state["model"] = self.backbone.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        if self.active_dataset is None:
            state['dataset'] = None
        else:
            state["dataset"] = self.active_dataset.state_dict()
        return state

    def set_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.active_dataset_settings = state_dict["dataset"]
        if self.active_dataset is not None:
            self.active_dataset.load_state_dict(self.active_dataset_settings)


class CalibratedActiveLearning(ActiveLearning):
    def train_on_loader(self, loader: DataLoader):
        self.wrapper.load_state_dict(self.initial_weights)
        if self.active_dataset is None:
            self.active_dataset = loader.dataset
            if self.active_dataset_settings is not None:
                self.active_dataset.load_state_dict(self.active_dataset_settings)
            self.loop.dataset = self.active_dataset

        hist, best_weight = self.wrapper.train_and_test_on_datasets(self.active_dataset,
                                                                    self.optimizer,
                                                                    self.batch_size,
                                                                    epoch=self.learning_epoch,
                                                                    use_cuda=True,
                                                                    return_best_weights=True,
                                                                    patience=5)

        self.wrapper.load_state_dict(best_weight)

        metrics = self.wrapper.metrics
        return self._format_metrics(metrics, 'train')

    def val_on_loader(self, loader, savedir=None):
        val_data = loader.dataset
        self.calibrator.calibrate(self.calib_set, self.valid_set,
                                  batch_size=16, epoch=10, use_cuda=True,
                                  double_fit=True)
        calibrated_model = ModelWrapper(self.calibrator.calibrated_model,
                                        None)
        if self.calibrate:
            self.loop.get_probabilities = calibrated_model.predict_on_dataset
        self.loop.step()
        self.wrapper.test_on_dataset(val_data, batch_size=self.batch_size, use_cuda=True)
        metrics = self.wrapper.metrics
        mets = self._format_metrics(metrics, 'test')
        mets.update({'num_samples': len(self.active_dataset)})
        return mets


def get_lambda(alpha, rng_state: np.random.RandomState):
    return rng_state.beta(alpha, alpha)


def get_mixup_params(input, alpha, rng, rng_numpy):
    lmb = get_lambda(alpha, rng_numpy)
    permutation = torch.randperm(input.size(0), generator=rng, device=input.device)
    return lmb, permutation


def patch_layer_mixup(module, name, alpha, seed):
    mods = dict(module.named_modules())
    if '.' in name:
        splitted = name.split('.')
        mod_name = splitted[0]
        name = '.'.join(splitted[1:])
        patch_layer_mixup(mods[mod_name], name, alpha, seed)
    else:
        l = mods[name]
        module.add_module(name, MixUpLayer(l, alpha, seed))

    return module


class MixUpLayer(nn.Module):
    def __init__(self, layer, alpha, seed):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.rng_numpy = np.random.RandomState(self.seed)

    def forward(self, input: torch.Tensor):
        if self.training:
            lmb, permutation = get_mixup_params(input, self.alpha, self.rng, self.rng_numpy)
            input_perm = input[permutation, ...]
            input = lmb * input + (1 - lmb) * input_perm
        out = self.layer(input)
        return out


class MixUpCriterion(MixUpLayer):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.training:
            lmb, permutation = get_mixup_params(input, self.alpha, self.rng, self.rng_numpy)
            target_perm = target[permutation]
            loss = lmb * self.layer(input, target)
            loss = loss + (1 - lmb) * self.layer(input, target_perm)
        else:
            loss = self.criterion(input, target)
        return loss


class MixUpActiveLearning(ActiveLearning):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)
        self.alpha = exp_dict.get('mixup_alpha', 0.0)
        self.mixup_seed = exp_dict.get('mixup_seed', 1337)
        self.mixup_layer_name = exp_dict.get('mixup_layer_name')

        self.criterion = MixUpCriterion(self.criterion, self.alpha, self.mixup_seed)

        if self.mixup_layer_name is not None:
            self.wrapper.model = patch_layer_mixup(self.wrapper.model,
                                                   self.mixup_layer_name,
                                                   self.alpha,
                                                   self.mixup_seed)
        else:
            warnings.warn('NO MIXUP', UserWarning)


class EmbeddingPropActiveLearning(ActiveLearning):
    # PAU YOUR CODE HERE
    pass
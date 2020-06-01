import numpy as np
import torch
from torch import nn


def get_lambda(alpha, rng_state: np.random.RandomState):
    return rng_state.beta(alpha, alpha)


def get_mixup_params(input, alpha, rng, rng_numpy):
    lmb = get_lambda(alpha, rng_numpy)
    permutation = torch.randperm(input.size(0), generator=rng, device=input.device)
    return lmb, permutation


def patch_layer_mixup(module, name, alpha, seed):
    """
    Patch a module by change layer with name `name`.

    Args:
        module (nn.Module): A Module
        name (str): Layer name (ex: features.2)
        alpha (float): Alpha threshold for Mixup
        seed (int): Seed for the MixUp sync.
    """
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
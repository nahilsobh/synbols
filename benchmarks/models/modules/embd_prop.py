import torch
from torch import nn

from utils.embedding_propagation import embedding_propagation


class EmbeddingPropLayer(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, input: torch.Tensor):
        assert (len(input.shape) == 2)
        embd_prop = embedding_propagation(input, alpha=0.5, rbf_scale=1, norm_prop=False)
        return self.l(embd_prop)


def patch_layer_embed_prop(module, name):
    mods = dict(module.named_modules())
    if '.' in name:
        splitted = name.split('.')
        mod_name = splitted[0]
        name = '.'.join(splitted[1:])
        print(f"Patching {mod_name}, {name}")
        patch_layer_embed_prop(mods[mod_name], name)
    else:
        l = mods[name]
        module.add_module(name, EmbeddingPropLayer(l))

    return module

import json
from typing import Dict

import numpy as np
from PIL import Image
from baal.active.heuristics.heuristics import _shuffle_subset
from skimage import img_as_float
from skimage.util import random_noise
from torchvision.transforms import Compose

from datasets.synbols import Synbols


class ColorNoise:
    def __init__(self, p, sigma, seed):
        self.p = p
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def __call__(self, x):
        if self.rng.rand() < self.p:
            x = Image.fromarray(random_noise(img_as_float(x), sigma=self.sigma ** 2))
        return x


SLANT_MAP = {
    'italic',
    'normal',
    'oblique'
}


class Attributes:
    def __init__(self, background='gradient', foreground='gradient',
                 slant=None, is_bold=None, rotation=None, scale=None, translation=None,
                 inverse_color=None,
                 pixel_noise_scale=0.01, resolution=(32, 32), rng=np.random.RandomState(42)):
        self.is_bold = rng.choice([True, False]) if is_bold is None else is_bold
        self.slant = rng.choice(list(SLANT_MAP)) if slant is None else slant
        self.background = background
        self.foreground = foreground
        self.rotation = rng.randn() * 0.2 if rotation is None else rotation
        self.scale = tuple(np.exp(rng.randn(2) * 0.1)) if scale is None else scale
        self.translation = tuple(rng.rand(2) * 0.2 - 0.1) if translation is None else translation
        self.inverse_color = rng.choice([True, False]) if inverse_color is None else inverse_color

        self.resolution = resolution
        self.pixel_noise_scale = pixel_noise_scale
        self.rng = rng

        # populated by make_image
        self.text_rectangle = None
        self.main_char_rectangle = None

    def to_dict(self):
        return dict(
            is_bold=str(self.is_bold),
            slant=self.slant,
            scale=self.scale,
            translation=self.translation,
            inverse_color=str(self.inverse_color),
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            text_rectangle=self.text_rectangle,
            main_char_rectangle=self.main_char_rectangle,
        )


class AleatoricSynbols(Synbols):
    def __init__(self, uncertainty_config: Dict, path, split, key='font', transform=None, p=0.0,
                 seed=None, n_classes=None, pixel_sigma=0, pixel_p=0):
        super().__init__(path=path, split=split, key=key, transform=transform)
        self.uncertainty_config = uncertainty_config
        self.p = p
        self.pixel_sigma = pixel_sigma
        self.pixel_p = pixel_p
        self.seed = seed
        self.noise_classes = n_classes
        self.rng = np.random.RandomState(self.seed)
        if self.pixel_p > 0 and split == 'train':
            self.transform = self._add_pixel_noise()
        if self.p > 0 and len(uncertainty_config) == 0:
            self.y = self._shuffle_label()
        elif self.p > 0:
            self._create_aleatoric_noise()

    def get_splits(self, source):
        if self.split == 'train':
            start = 0
            end = int(0.7 * len(source))
        if self.split == 'calib':
            start = int(0.7 * len(source))
            end = int(0.8 * len(source))
        elif self.split == 'val':
            start = int(0.8 * len(source))
            end = int(0.9 * len(source))
        elif self.split == 'test':
            start = int(0.9 * len(source))
            end = len(source)
        return start, end

    def get_values_split(self, y):
        start, end = self.get_splits(source=y)
        return y[self.indices[start:end]]

    def _shuffle_label(self):
        return _shuffle_subset(self.y, self.p)

    def _add_pixel_noise(self):
        return Compose([ColorNoise(self.p, self.pixel_sigma, self.seed), self.transform])

    def _create_aleatoric_noise(self):
        self._latent_space = self._get_latent_space()
        if self.noise_classes is not None:
            clss = self.rng.choice(self.num_classes, self.noise_classes)
        else:
            clss = np.arange(self.num_classes)
        data = np.load(self.path)
        y = data['y']
        del data
        _y = []
        for yi in y:
            j = json.loads(yi)
            _y.append({key: j[key] for key in self.uncertainty_config.keys()})

        y_flag = [self._isin_latent(yi) for yi in _y]
        y_flag = self.get_values_split(np.array(y_flag))
        assert len(y_flag) == len(self.y)
        print(f"{sum(y_flag)} items are in the latent space out of {len(y_flag)}.")
        y_flag = [True if yi and self.rng.rand() < self.p else False for yi in y_flag]
        y_flag = np.logical_and(y_flag, np.isin(self.y, clss))
        targets = self.y[y_flag]
        print(f"{len(targets)} items will be shuffled in {len(clss)} classes.")
        print(f"Classes with some elements shuffled: {np.unique(targets)}")
        self.rng.shuffle(targets)
        self.y[y_flag] = targets

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)

    def _get_latent_space(self) -> Dict:
        return Attributes(rng=self.rng).to_dict()

    def _isin_latent(self, y):
        """Returns a bitmap of any"""
        flag = True
        for key, val in y.items():
            latent_val = self._latent_space[key]
            if isinstance(val, float):
                scale = self.uncertainty_config[key].get('scale', 0.02)
                flag = flag and (latent_val - scale) < val < (latent_val + scale)
            else:
                flag = flag and (val == latent_val)
        return flag


if __name__ == '__main__':
    synbols = AleatoricSynbols(uncertainty_config={'is_bold': {}},
                               p=0.05,
                               key='char',
                               n_classes=5,
                               path='/mnt/datasets/public/research/synbols/old/latin_res=32x32_n=100000.npz',
                               split='train')

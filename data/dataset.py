import os
import random

import torch.utils.data as data
import torchvision.transforms.functional as ttf
from PIL import Image
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor


class BaseDataset(data.Dataset):
    def __init__(self, opts):
        self.base = opts.base
        self.transforms = None
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        w, h = img.size
        base = self.base
        pad_h, pad_w = (base - h % base) % base, (base - w % base) % base
        img = ttf.pad(img, padding=[0, 0, pad_w, pad_h], padding_mode='edge')
        if self.transforms:
            img = self.transforms(img)
        return img, pad_h, pad_w


class SingleDataset(BaseDataset):
    def __init__(self, opts):
        super().__init__(opts)
        self.test_path = opts.test_path
        self.imgs = [os.path.join(opts.test_path, img) for img in os.listdir(self.test_path)]
        self.transforms = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        print(f'total: {len(self.imgs)} images')

    def __getitem__(self, index):
        img, pad_h, pad_w = self.load_img(self.imgs[index])
        return [img, pad_h, pad_w, self.imgs[index]]

    def __len__(self):
        return len(self.imgs)


class PairDataset(BaseDataset):
    def __init__(self, opts):
        super().__init__(opts)
        self.val_path = opts.val_path

        lq_dir = os.path.join(self.val_path, 'input')
        self.lq_paths = sorted([os.path.join(lq_dir, x) for x in os.listdir(lq_dir)])
        hq_dir = os.path.join(self.val_path, 'target')
        self.hq_paths = sorted([os.path.join(hq_dir, x) for x in os.listdir(hq_dir)])
        assert len(self.lq_paths) == len(self.hq_paths), ' lq and hq have different number of images'

        self.transforms = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        print(f'dataset size: {len(self.lq_paths)}')

    def __getitem__(self, index):
        lq = self.load_img(self.lq_paths[index])
        hq = self.load_img(self.hq_paths[index])
        return lq, hq

    def __len__(self):
        return len(self.lq_paths)


class UnpairedPatchesDataset(BaseDataset):
    def __init__(self, opts):
        super().__init__(opts)
        self.train_path = opts.train_path

        lq_dir = os.path.join(self.train_path, 'input')
        self.lq_paths = [os.path.join(lq_dir, x) for x in os.listdir(lq_dir)]
        hq_dir = os.path.join(self.train_path, 'target')
        self.hq_paths = [os.path.join(hq_dir, x) for x in os.listdir(hq_dir)]

        self.lq_size = len(self.lq_paths)
        self.hq_size = len(self.hq_paths)
        self.dataset_size = max(self.lq_size, self.hq_size) if not opts.is_debug else 8  # for debug
        assert opts.patch_size % self.base == 0, 'patch size should be multiple of base'
        self.patch_size = opts.patch_size

        # setup image transformation
        transforms = []
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('lq: %d, hq: %d images' % (self.lq_size, self.hq_size))
        return

    def __getitem__(self, index):
        data_A = self.load_img(self.lq_paths[index % self.lq_size])
        data_B = self.load_img(self.hq_paths[random.randint(0, self.hq_size - 1)])
        patch_a0, patch_a1 = self._crop_patch(data_A), self._crop_patch(data_A)
        patch_b0, patch_b1 = self._crop_patch(data_B), self._crop_patch(data_B)
        return [patch_a0, patch_a1], [patch_b0, patch_b1]

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        if img.size[0] < self.patch_size or img.size[1] < self.patch_size:
            img = img.resize((int(self.patch_size * 1.5), int(self.patch_size * 1.5)), Image.BICUBIC)
        img = self.transforms(img)
        return img

    def _crop_patch(self, img):
        h, w = img.shape[1:]
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        return img[:, x : x + self.patch_size, y : y + self.patch_size]

    def __len__(self):
        return self.dataset_size

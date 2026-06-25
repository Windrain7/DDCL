import os

import torch
import tqdm

from data.dataset import SingleDataset
from models.ddcl import DDCL
from utils.options import TestOptions
from utils.saver import save_img


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    opts.is_test = True

    # data loader
    print('\n--- load dataset ---')
    dataset = SingleDataset(opts)

    # model
    print('\n--- load model ---')
    model = DDCL(opts)
    # set the avalible gpu in the device
    gpu = torch.cuda.current_device()
    model.setgpu(gpu)
    ep, _ = model.resume(opts.resume, train=False)
    print(f'Model was successfully loaded from epoch {ep}.')

    # 适配分布式训练所得权重
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # test
    print('\n--- testing ---')
    for img, pad_h, pad_w, path in tqdm.tqdm(dataset, desc='Testing', unit='image'):
        with torch.no_grad():
            model.feed_data(img[None, :], img[None, :])
            img = model.test_forward()
            mask = model.mask_a_0
        img, mask = img.squeeze(), mask.squeeze()
        if pad_h > 0:
            img, mask = img[:, :-pad_h, :], mask[:, :-pad_h, :]
        if pad_w > 0:
            img, mask = img[:, :, :-pad_w], mask[:, :, :-pad_w]
        filename = os.path.basename(path)
        save_img(img, os.path.join(result_dir, filename))
        if opts.save_mask:
            base, ext = os.path.splitext(filename)
            save_img(mask, os.path.join(result_dir, f'{base}_mask{ext}'))
    return


if __name__ == '__main__':
    main()

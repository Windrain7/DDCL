import os
from datetime import datetime

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# tensor to PIL Image
def tensor2img(img):
    img = torch.clamp(img, -1.0, 1.0)
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    # pdb.set_trace()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    # img = (np.transpose(img, (1, 2, 0))) * 255.0
    return img.astype(np.uint8)


def save_img(img, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path)


class Saver:
    def __init__(self, opts):
        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.ckpt_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.ckpt_dir, 'images')
        self.dict_dir = os.path.join(self.ckpt_dir, 'dicts')
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        self.saved_cnt, self.max_save_count = 0, 3
        self.saved_psnr_ssim_ep = [(0, 0, 0)]

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.dict_dir):
            os.makedirs(self.dict_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and attr.startswith('loss')]
            for m in members:
                loss_value = getattr(model, m).item()
                self.writer.add_scalar(m, loss_value, total_it)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0 or ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen&mask_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.ckpt_dir, ep), ep, total_it)
        elif ep == -1:
            model.save(f'{self.ckpt_dir}/last.pth', ep, total_it)

    # save dict
    def write_dict(self, obj, ep, model):
        if (ep + 1) % self.model_save_freq == 0:
            dict_filename = '%s/%05d' % (self.dict_dir, ep)
            print('--- save the dict @ ep %d ---' % (ep))
            model.save_dict(obj, dict_filename)
        elif ep == -1:
            dict_filename = f'{self.dict_dir}/last'
            model.save_dict(dict, dict_filename)
            dict_filename = f'{self.dict_dir}/last'
            model.save_dict(dict, dict_filename)

    # save best model
    def write_best_model(self, ep, total_it, model, psnr_avg, ssim_avg):
        metric_info = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{ep:03d}\t{psnr_avg:.4f}\t{ssim_avg:.4f}"

        def get_model_path(index):
            return os.path.join(self.ckpt_dir, f'net{index + 1}.pth')

        if self.saved_cnt == 0:
            model.save(get_model_path(0), ep, total_it)
            print(f'--- save the model @ {ep} as net1.pth ---')
            metric_info += '\tsave as net1.pth'
            self.saved_cnt += 1
            self.saved_psnr_ssim_ep[0] = (psnr_avg, ssim_avg, ep)
            return metric_info

        for i, (psnr, ssim, _) in zip(range(self.saved_cnt), self.saved_psnr_ssim_ep):
            if ssim_avg * 10 + psnr_avg >= ssim * 10 + psnr:
                # 如果已满，删除最旧的模型文件
                if self.saved_cnt == self.max_save_count:
                    os.remove(get_model_path(self.max_save_count - 1))
                    self.saved_cnt -= 1

                # 将文件向后移动一位，腾出空位
                for j in range(self.saved_cnt - 1, i - 1, -1):
                    os.rename(get_model_path(j), get_model_path(j + 1))

                # 保存新的模型文件
                model.save(get_model_path(i), ep, total_it)
                print(f'--- save the model @ {ep} as net{i + 1}.pth ---')
                metric_info += f'\tsave as net{i + 1}.pth'

                self.saved_cnt += 1
                # 更新保存的性能指标列表
                self.saved_psnr_ssim_ep.insert(i, (psnr_avg, ssim_avg, ep))
                if len(self.saved_psnr_ssim_ep) > self.max_save_count:
                    self.saved_psnr_ssim_ep.pop()
                break

        metric_info += '\t' + '\t'.join(f'net{i+1}: {x[2]}' for i, x in enumerate(self.saved_psnr_ssim_ep))
        return metric_info

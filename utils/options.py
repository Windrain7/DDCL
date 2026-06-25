import argparse
import os
import random

from .utils import set_seed


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--seed', type=int, default=666, help='random seed')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--is_test', action='store_true', default=False, help='train or test')
        self.parser.add_argument('--base', type=int, default=8, help='the base of the image to be modified')
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')

        # model
        self.parser.add_argument('--discriminator', type=str, default='MultiScaleDis', help='the type of discriminator')
        self.parser.add_argument('--dis_FFT_mode', type=int, default=0, help='the mode of FFT in discriminator')
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')

        self.parser.add_argument('--moco_FFT_mode', type=int, default=1, help='the mode of moco')
        self.parser.add_argument('--moco_num_layers', type=int, nargs='+', default=[1, 1, 1], help='the number of layers in moco')
        self.parser.add_argument('--moco_latent_num', type=int, default=1, help='the number of layers in the latent layer of moco')
        self.parser.add_argument('--moco_dim', default=256, type=int, help='the dimension of moco')
        self.parser.add_argument('--moco_K', default=8192, type=int, help='the size of queue in moco')

    def parse(self):
        opt = self.parser.parse_args()
        if opt.seed is None:
            opt.seed = random.randint(1, 10000)
        set_seed(opt.seed)
        self.print_options(opt)
        return opt

    def print_options(self, opt):
        """Print and save options"""
        message = ''
        message += '-' * 20 + 'Options' + '-' * 20 + '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {str(default)}]'
            message += f'{str(k):>25}: {str(v):<30}{comment}\n'
        message += '-' * 20 + 'Options' + '-' * 20 + '\n'
        print(message)

        # save to the disk
        if not os.path.exists(os.path.join(opt.result_dir, opt.name)):
            os.makedirs(os.path.join(opt.result_dir, opt.name))
        file_name = os.path.join(opt.result_dir, opt.name, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

        # data loader related
        self.parser.add_argument('--train_path', type=str, default='', help='path of training data')
        self.parser.add_argument('--val_path', type=str, default='', help='path of testing data')
        self.parser.add_argument('--is_debug', action='store_true', default=False, help='to debug')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--patch_size', type=int, default=216, help='patch size for training')
        self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
        self.parser.add_argument('--no_flip', action='store_true', default=False, help='specified if no flipping')

        # ouptput related
        self.parser.add_argument('--display_dir', type=str, default='tb_logger', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='experiments', help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=100, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

        # training related
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=400, help='number of epochs')
        self.parser.add_argument('--n_ep_decay', type=int, default=200, help='epoch start decay learning rate, set -1 if no decay')
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')

        # loss
        self.parser.add_argument(
            '--gan_mode',
            type=str,
            default='lsgan',
            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.',
        )
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--GAN_lambda', type=float, default=1, help='the weight of GAN loss')

        self.parser.add_argument('--low_th', type=float, default=0.1, help='the low frequency threshold of FreqMagLoss')
        self.parser.add_argument('--crs_th', type=float, default=0.03, help='the threshold of cross size in FreqMagLoss')
        self.parser.add_argument('--std_th', type=float, default=1.5, help='the standard deviation threshold of FreqMagLoss')
        self.parser.add_argument('--window_size', type=int, default=53, help='the window size of average pooling in FreqMagLoss')
        self.parser.add_argument(
            '--extract_out', action='store_false', default=True, help="whether to extract the top value from the model's output in FreqMagLoss"
        )
        self.parser.add_argument('--freq_lambda', type=float, default=1.0, help='the weight of FreqMagloss')

        self.parser.add_argument('--FFT_loss', action='store_true', default=False, help='whether to use FFTLoss')
        self.parser.add_argument('--FFT_lambda', type=float, default=0.1, help='the weight of FFTLoss')

        self.parser.add_argument('--sec_mask_loss', action='store_false', help='add the second mask loss')
        self.parser.add_argument('--cycle_lambda', type=float, default=10, help='the weight of cycle loss')
        self.parser.add_argument('--perceptual_lambda', type=float, default=0.01, help='the weight of perceptual loss')
        self.parser.add_argument(
            '--mask_combine', type=str, default='add', choices=['add', 'brightness', 'screen'], help='the way to combine mask and background'
        )
        self.parser.add_argument('--mask_lambda', type=float, default=10, help='the weight of mask loss')
        self.parser.add_argument('--contrast_lambda', type=float, default=0.1, help='the weight of contrast loss')

    def parse(self):
        opt = super().parse()
        if opt.is_debug:
            opt.nThreads = 0
        return opt


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()

        # data loader related
        self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')

        # output related
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--save_mask', action='store_true', help='specified if saving the mask')
        self.parser.add_argument('--result_dir', type=str, default='outputs', help='path for saving result images and models')

        # model related
        self.parser.add_argument('--resume', type=str, required=True, help='specified path of saved model to load')
        self.parser.add_argument('--test_path', type=str, default='datasets', help='path of testing data')

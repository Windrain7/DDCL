import os

import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataset import PairDataset, UnpairedPatchesDataset
from metrics.metrics import calculate_psnr, calculate_ssim
from models.ddcl import DDCL
from utils.options import TrainOptions
from utils.saver import Saver
from utils.utils import *


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # Initialize distributed training
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        is_distributed = True
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # data loader
    print('\n--- load dataset ---')
    dataset = UnpairedPatchesDataset(opts)
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads, drop_last=True, sampler=train_sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads, drop_last=True)

    dataset_val = PairDataset(opts)
    if is_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=opts.nThreads, drop_last=False, sampler=val_sampler
        )
    else:
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=opts.nThreads, drop_last=False)

    if not is_distributed or dist.get_rank() == 0:
        if not os.path.exists(os.path.join(opts.result_dir, opts.name)):
            os.makedirs(os.path.join(opts.result_dir, opts.name))
        trainLogger = open(f'{os.path.join(opts.result_dir, opts.name)}/psnr_ssim.log', 'a')
        lossLogger = open(f'{os.path.join(opts.result_dir, opts.name)}/train_loss.log', 'a')

    # model
    print('\n--- load model ---')
    model = DDCL(opts)
    model.setgpu(local_rank)
    model.to(device)  # Move model to GPU before wrapping with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    if opts.resume:
        ep0, total_it = model.module.resume(opts.resume) if is_distributed else model.resume(opts.resume)
    elif pth_files := [f for f in os.listdir(os.path.join(opts.result_dir, opts.name)) if os.path.splitext(f)[0].isdigit()]:
        # try auto resume
        max_pth_file = max(pth_files, key=lambda f: int(os.path.splitext(f)[0]))
        ep0, total_it = (
            model.module.resume(os.path.join(opts.result_dir, opts.name, max_pth_file))
            if is_distributed
            else model.resume(os.path.join(opts.result_dir, opts.name, max_pth_file))
        )
    else:
        if is_distributed:
            model.module.initialize()
        else:
            model.initialize()
        ep0 = -1
        total_it = 0

    if is_distributed:
        model.module.set_scheduler(opts, last_ep=ep0)
    else:
        model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    if not is_distributed or dist.get_rank() == 0:
        saver = Saver(opts)

    # train
    print('\n--- train ---')
    for ep in range(ep0, opts.n_ep):
        if is_distributed:
            train_sampler.set_epoch(ep)
        for it, (patches_a, patches_b) in enumerate(train_loader):
            # input data
            for i in range(len(patches_a)):
                patches_a[i] = patches_a[i].to(device)
            for i in range(len(patches_b)):
                patches_b[i] = patches_b[i].to(device)
            if is_distributed:
                model.module.feed_data(patches_a, patches_b)
                model.module.update_EG(ep, opts)
                model.module.update_D(opts)
            else:
                model.feed_data(patches_a, patches_b)
                model.update_EG(ep, opts)
                model.update_D(opts)

            # save to display file
            if (not is_distributed or dist.get_rank() == 0) and not opts.no_display_img:
                saver.write_display(total_it, model.module if is_distributed else model)

            if (not is_distributed or dist.get_rank() == 0) and total_it % opts.display_freq == 0:
                loss_info = (
                    f'total_it: {total_it} [ep {ep}, it {it}]\tlr {model.module.genA_opt.param_groups[0]["lr"]:.08f}'
                    if is_distributed
                    else f'total_it: {total_it} [ep {ep}, it {it}]\tlr {model.genA_opt.param_groups[0]["lr"]:.08f}'
                )
                members = [
                    attr
                    for attr in dir(model.module if is_distributed else model)
                    if not callable(getattr(model.module if is_distributed else model, attr)) and attr.startswith('loss')
                ]
                for m in members:
                    loss_value = getattr(model.module if is_distributed else model, m).item()
                    loss_info += f', {m}: {loss_value:.04f}'
                print(loss_info)
                lossLogger.write(f'{loss_info}\n')
                lossLogger.flush()
            total_it += 1

        # Save network weights
        if not is_distributed or dist.get_rank() == 0:
            saver.write_model(ep, total_it, model.module if is_distributed else model)
            # save result image
            saver.write_img(ep, model.module if is_distributed else model)

        # decay learning rate
        if opts.n_ep_decay > -1:
            if is_distributed:
                model.module.update_lr()
            else:
                model.update_lr()

        print('\n--- valing ---')
        ssim_avg, psnr_avg = 0, 0
        for input_val, target_val in tqdm.tqdm(loader_val, desc='Valing', unit='image'):
            _, pad_h, pad_w = input_val
            pad_h, pad_w = int(pad_h), int(pad_w)
            input_val, target_val = input_val[0].to(device), target_val[0].to(device)
            with torch.no_grad():
                if is_distributed:
                    model.module.feed_data(input_val, target_val)
                    out_val = model.module.test_forward()
                else:
                    model.feed_data(input_val, target_val)
                    out_val = model.test_forward()
                # note: if pad_h == 0, out_val[:, :, :-pad_h, :] will be zero
                if pad_h > 0:
                    out_val, target_val = out_val[:, :, :-pad_h, :], target_val[:, :, :-pad_h, :]
                if pad_w > 0:
                    out_val, target_val = out_val[:, :, :, :-pad_w], target_val[:, :, :, :-pad_w]
                out_val = (torch.clamp(out_val, -1.0, 1.0) + 1) / 2
                target_val = (torch.clamp(target_val, -1.0, 1.0) + 1) / 2

                ssim_val = calculate_ssim(out_val * 255, target_val * 255, 0, 'CHW', True)
                psnr_val = calculate_psnr(out_val, target_val, 0, 'CHW', True)
                ssim_avg += ssim_val
                psnr_avg += psnr_val

        ssim_avg /= len(loader_val)
        psnr_avg /= len(loader_val)
        print(f'[epoch {ep}] psnr: {psnr_avg:.2f}, ssim: {ssim_avg:.4f}')
        if not is_distributed or dist.get_rank() == 0:
            saver.writer.add_scalar('psnr', psnr_avg, ep)
            saver.writer.add_scalar('ssim', ssim_avg, ep)

            metric_info = saver.write_best_model(ep, total_it, model.module if is_distributed else model, psnr_avg, ssim_avg)
            trainLogger.write(f'{metric_info}\n')
            trainLogger.flush()

    if not is_distributed or dist.get_rank() == 0:
        trainLogger.close()
        lossLogger.close()

    return


if __name__ == '__main__':
    main()

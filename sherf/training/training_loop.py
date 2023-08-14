# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import torch.distributed as dist

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section
from training.test_loop import test 

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

#----------------------------------------------------------------------------

def to_cuda(device, sp_input):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = sp_input[key].to(device)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = sp_input['params'][key1].to(device)
    
        if key=='t_params':
            for key1 in sp_input['t_params']:
                if torch.is_tensor(sp_input['t_params'][key1]):
                    sp_input['t_params'][key1] = sp_input['t_params'][key1].to(device)

        if key=='obs_params':
            for key1 in sp_input['obs_params']:
                if torch.is_tensor(sp_input['obs_params'][key1]):
                    sp_input['obs_params'][key1] = sp_input['obs_params'][key1].to(device)

    return sp_input


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    recons_loss             = True,
    use_sr_module           = True,
    cfg                     = 'THuman',
    test_flag               = False,
):
    # Initialize.
    start_time = time.time()
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        # print('Image shape:', training_set.image_shape)
        # print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    # common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    common_kwargs = dict(c_dim=0, img_resolution=512, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)

        # G = copy.deepcopy(resume_data['G']).train().requires_grad_(False).to(device)
        # D = copy.deepcopy(resume_data['D']).train().requires_grad_(False).to(device)
        # G_ema = copy.deepcopy(resume_data['G_ema']).eval().requires_grad_(False).to(device)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=True)

    # Print network summary tables.
    # if rank == 0:
    #     z = torch.empty([batch_gpu, G.z_dim], device=device)
    #     c = torch.empty([batch_gpu, G.c_dim], device=device)
    #     img = misc.print_module_summary(G, [z, c])
    #     misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
    # for module in [G, G_ema]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
    # for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20000//batch_size, gamma=0.5)
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, scheduler=scheduler, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, scheduler=scheduler, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    # grid_size = None
    # grid_z = None
    # grid_c = None
    # if rank == 0:
    #     print('Exporting sample images...')
    #     grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    #     save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
    #     grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
    #     grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    running_loss_Gmain_train = running_img_loss_raw_train = running_acc_loss_raw_train \
        = running_ssim_raw_train = running_lpips_raw_train \
            = running_img_loss_train = running_loss_Dgen = running_loss_Gmain_Dgen \
                = running_loss_Dr1 = running_loss_Dreal = 0

    loss_Dr1 = None
    loss_Dr1_count = 0

    if test_flag:
        if rank == 0:
            testsavedir = os.path.join(run_dir, 'testset_{:06d}_more'.format(cur_nimg))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
            # test_THuman(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
            # test_zju_mocap(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
            # test_HuMMan(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, smc_file=training_set_kwargs.data_root)
            # test_HuMMan(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
                if cfg == 'RenderPeople':
                    test(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, dataset_name=cfg, data_root=training_set_kwargs.data_root, obs_view_lst=[0, 16, 31], nv_pose_start=0, np_pose_start=2, pose_interval=2, pose_num=5)
                elif cfg == 'THuman':
                    test(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, dataset_name=cfg, data_root=training_set_kwargs.data_root, obs_view_lst=[4, 12, 20], nv_pose_start=0, np_pose_start=0, pose_interval=2, pose_num=5)       
                elif cfg == 'HuMMan':
                    test(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, dataset_name=cfg, data_root=training_set_kwargs.data_root, obs_view_lst=[0, 4, 8], nv_pose_start=0, np_pose_start=0, pose_interval=6, pose_num=17)  
                elif cfg == 'zju_mocap':
                    test(G, savedir=testsavedir, neural_rendering_resolution=loss_kwargs['neural_rendering_resolution_initial'], rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, dataset_name=cfg, data_root=training_set_kwargs.data_root, obs_view_lst=[4, 10, 16], nv_pose_start=0, np_pose_start=0, pose_interval=20, pose_num=25)                                  
        exit(0)

    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            input_data = next(training_set_iterator)
            input_data = to_cuda(device, input_data)
            phase_real_img = input_data['img_all'][:,0].to(device).to(torch.float32).split(batch_gpu)
            phase_real_c = torch.zeros(1, 25).to(device).split(batch_gpu)
            # all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            # all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [np.zeros((1, 25)) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]   

            # phase_real_img, phase_real_c = next(training_set_iterator)
            # phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            # phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            # all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            # all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            # all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                if phase.name in ['Gmain', 'Gboth']:
                    loss_Gmain_train, img_loss_raw_train, acc_loss_raw_train, ssim_raw_train, lpips_raw_train, loss_Gmain_Dgen = loss.accumulate_gradients(phase=phase.name, input_data=input_data, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, use_sr_module=use_sr_module, recons_loss=recons_loss, rank=rank)
                elif phase.name in ['Dmain', 'Dboth']:
                    loss_Dgen, loss_Dreal = torch.zeros(1).to(device), torch.zeros(1).to(device)
                elif phase.name in ['Dreg', 'Dboth']:
                    loss_Dr1 = torch.zeros(1).to(device)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
                phase.scheduler.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            # ema_beta = 0.00002
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        running_loss_Gmain_train += loss_Gmain_train
        running_img_loss_raw_train += img_loss_raw_train
        running_acc_loss_raw_train += acc_loss_raw_train
        running_ssim_raw_train += ssim_raw_train
        running_lpips_raw_train += lpips_raw_train        
        running_loss_Gmain_Dgen += loss_Gmain_Dgen
        running_loss_Dgen += loss_Dgen
        running_loss_Dreal += loss_Dreal
        if loss_Dr1 is not None:
            loss_Dr1_count += 1
            running_loss_Dr1 += loss_Dr1.clone()
            loss_Dr1 = None

        # Evaluate metrics.
        i_print = 100 #batch_size*100
        if cur_nimg % i_print == 0 and cur_nimg > 0:
            if num_gpus > 1:
                dist.all_reduce(running_loss_Gmain_train)
                dist.all_reduce(running_img_loss_raw_train)
                dist.all_reduce(running_acc_loss_raw_train)
                dist.all_reduce(running_ssim_raw_train)
                dist.all_reduce(running_lpips_raw_train)
                dist.all_reduce(running_loss_Gmain_Dgen)
                dist.all_reduce(running_loss_Dgen)
                dist.all_reduce(running_loss_Dreal)
                dist.all_reduce(running_loss_Dr1)
            running_loss_Gmain_train /= i_print
            running_img_loss_raw_train /= i_print
            running_acc_loss_raw_train /= i_print
            running_ssim_raw_train /= i_print
            running_lpips_raw_train /= i_print
            # running_img_loss_train /= i_print
            running_loss_Gmain_Dgen /= i_print
            running_loss_Dgen /= i_print
            running_loss_Dreal /= i_print
            running_loss_Dr1 /= (loss_Dr1_count*num_gpus)
            if rank == 0:
                psnr_raw = mse2psnr(running_img_loss_raw_train)
                stats_metrics['psnr_raw'] = round(psnr_raw.item(), 3)
                print("[TRAIN] Cur_nimg: {}  Loss: {} Lr: {} Img Loss raw: {} Acc Loss raw {} SSIM raw {} Lpips raw {} loss Gmain Dgen {} Dgen Loss {} Dreal Loss {} Dr1 Loss {} PSNR raw: {}".\
                    format(cur_nimg, round(running_loss_Gmain_train.item(), 5), round(opt.param_groups[0]['lr'], 5), round(running_img_loss_raw_train.item(), 5), \
                        round(running_acc_loss_raw_train.item(), 5), round(running_ssim_raw_train.item(), 3), round(running_lpips_raw_train.item(), 3), \
                        round(running_loss_Gmain_Dgen.item(), 5), round(running_loss_Dgen.item(), 5), round(running_loss_Dreal.item(), 5), \
                                round(running_loss_Dr1.item(), 5), round(psnr_raw.item(), 3)))           
                
        if cur_nimg % i_print == 0: 
            running_loss_Gmain_train = running_img_loss_raw_train = running_acc_loss_raw_train \
                = running_ssim_raw_train = running_lpips_raw_train \
                = running_img_loss_train = running_loss_Gmain_Dgen = running_loss_Dgen = \
                    running_loss_Dreal = running_loss_Dr1 = 0
            
            loss_Dr1 = None
            loss_Dr1_count = 0

        # i_test = 40000 #batch_size*1000
        # if (cur_nimg % i_test == 0 and cur_nimg > 0): #or cur_nimg == 4000:
        #     if rank == 0:
        #         testsavedir = os.path.join(run_dir, 'testset_{:06d}_more'.format(cur_nimg))
        #         os.makedirs(testsavedir, exist_ok=True)
        #         with torch.no_grad():
        #             if cfg == 'THuman':
        #                 test_THuman(G, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=G_kwargs.rendering_kwargs['white_back'], sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
        #             elif cfg == 'zju_mocap':
        #                 test_zju_mocap(G, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
        #             elif cfg == 'HuMMan':
        #                 # test_HuMMan(G, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, smc_file=training_set_kwargs.data_root)
        #                 test_HuMMan(G, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
        #             elif cfg == 'RenderPeople':
        #                 test_RenderPeople(G, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)

                # testsavedir = os.path.join(run_dir, 'testset_{:06d}_ema_more'.format(cur_nimg))
                # os.makedirs(testsavedir, exist_ok=True)
                # with torch.no_grad():
                #     if cfg == 'THuman':
                #         test_THuman(G_ema, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
                #     elif cfg == 'zju_mocap':
                #         test_zju_mocap(G_ema, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view)
                #     elif cfg == 'HuMMan':
                #         test_HuMMan(G_ema, savedir=testsavedir, neural_rendering_resolution=G.neural_rendering_resolution, rank=0, use_sr_module=use_sr_module, white_back=False, sample_obs_view=training_set_kwargs.sample_obs_view, fix_obs_view=training_set_kwargs.fix_obs_view, smc_file=training_set_kwargs['data_root'])

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 5000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))


        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        #     out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
        #     images = torch.cat([o['image'].cpu() for o in out]).numpy()
        #     images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
        #     images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
        #     save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
        #     save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
        #     save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        network_snapshot_ticks = 1
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
            # for name, module in [('G', G), ('G_ema', G_ema)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------

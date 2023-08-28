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
import imageio
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import torch.distributed as dist
import cv2
from skimage.metrics import structural_similarity

import legacy
from torch.utils.data import DataLoader
import lpips

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

loss_fn_vgg = lpips.LPIPS(net='vgg')

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

def ssim_metric(rgb_pred, rgb_gt, mask_at_box, H, W):
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt

    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    
    # compute the ssim
    ssim = structural_similarity(img_pred, img_gt, multichannel=True)
    lpips = loss_fn_vgg(torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.float32), torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.float32)).reshape(-1).item()

    return ssim, lpips

#----------------------------------------------------------------------------

def test(model, savedir=None, neural_rendering_resolution=128, rank=0, use_sr_module=False, white_back=False, sample_obs_view=False, fix_obs_view=False, dataset_name='RenderPeople', data_root=None, obs_view_lst = [0, 16, 31], nv_pose_start=0, np_pose_start=2, pose_interval=0, pose_num=5):

    device = torch.device('cuda', rank)
    batch_size = 1
    humans_data_root = os.path.dirname(data_root)

    ## novel view synthesis evaluation with obs image from the same pose
    pose_start = nv_pose_start # 0 
    pose_interval = pose_interval
    pose_num = pose_num
    data_interval = 2
    obs_view_lst = obs_view_lst #[0, 16, 31]

    humans_list = os.path.join(humans_data_root, 'human_list.txt')

    if dataset_name == 'RenderPeople':
        class_name = 'training.RenderPeople_dataset.RenderPeopleDatasetBatch'
        image_scaling=neural_rendering_resolution/512
        with open(humans_list) as f:
            humans_name = f.readlines()[450:480]
    elif dataset_name == 'THuman':
        class_name = 'training.THuman_dataset.THumanDatasetBatch'
        image_scaling=neural_rendering_resolution/512
        with open(humans_list) as f:
            humans_name = f.readlines()[90:100]
    elif dataset_name == 'HuMMan':
        class_name = 'training.HuMMan_dataset.HuMManDatasetBatch'
        image_scaling=1/3
        data_interval=1
        # with open(humans_list) as f:
        #     humans_name = f.readlines()[317:339]
        humans_name = [
            'p000455_a000986',
            'p000456_a000396',
            'p000465_a000048',
            'p000465_a000701',
            'p000474_a000048',
            'p000477_a000396',
            'p000482_a000793',
            'p000491_a005730',
            'p000503_a000064',
            'p000503_a000224',
            'p000532_a005711',
            'p000538_a000978',
            'p000538_a000986',
            'p000542_a000048',
            'p000545_a000064',
            'p000547_a000011',
            'p000547_a000145',
            'p000557_a000793',
            'p000582_a000048',
            'p100050_a001425',
            'p100056_a000049',
            'p100074_a000048',
        ]
    elif dataset_name == 'zju_mocap':
        class_name = 'training.NeuBody_dataset.NeuBodyDatasetBatch'
        image_scaling=0.5
        # with open(humans_list) as f:
        #     humans_name = f.readlines()[317:339]
        humans_name = [
            "CoreView_377", 
            "CoreView_313", 
            "CoreView_315"  
        ]

    all_humans = [os.path.join(humans_data_root, human_name.strip()) for human_name in humans_name]

    for obs_view in obs_view_lst:

        total_psnr = []
        total_ssim = []
        total_lpips = []

        for p, human_data_path in enumerate(all_humans):
            data_root = human_data_path.strip()
            human_name = os.path.basename(data_root)

            savedir_human = os.path.join(savedir, 'novel_view', f'obs_view_{obs_view}', human_name)
            os.makedirs(savedir_human, exist_ok=True)

            test_dataset_kwargs = dnnlib.EasyDict(class_name=class_name, data_root=data_root, split='test', multi_person=False, num_instance=1, poses_start=pose_start, poses_interval=pose_interval, poses_num=pose_num, image_scaling=image_scaling, white_back=white_back, sample_obs_view=sample_obs_view, fix_obs_view=fix_obs_view)
            test_set = dnnlib.util.construct_class_by_name(**test_dataset_kwargs) # subclass of training.dataset.Dataset
            test_set.obs_view_index = obs_view

            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

            psnr_sub_view = []
            ssim_sub_view = []
            lpips_sub_view = []

            for k, test_data in enumerate(test_loader):

                view_id = k % test_set.camera_view_num

                if view_id == obs_view or view_id % data_interval != 0:
                    continue

                print("novel view subject: ", human_name, " obs_view: ", obs_view)

                test_data = to_cuda(device, test_data)

                gen_img = model(test_data, torch.randn(1, 512).to(device), torch.zeros((1, 25)).to(device), \
                    neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, test_flag=True)

                H, W = test_data['img_all'].shape[-2], test_data['img_all'].shape[-1]
                mask_at_box = test_data['mask_at_box_large_all'].reshape(test_data['mask_at_box_large_all'].shape[0], H, W)

                for j in range(batch_size):

                    img_pred = (gen_img['image'][j] / 2 + 0.5).permute(1,2,0)
                    real_img = test_data['img_all'][j]
                    gt_img = real_img[j].permute(1,2,0)

                    rgb8 = to8b(img_pred.cpu().numpy())
                    gt_rgb8 = to8b(gt_img.cpu().numpy())
                    rgb8 = np.concatenate([rgb8, gt_rgb8], axis=1)

                    filename = os.path.join(savedir_human, '{:02d}_{:02d}_{:02d}.png'.format(int(test_data['pose_index'][j]), int(test_data['pose_index'][j]), view_id))

                    input_img = test_data['obs_img_all'][0,0].cpu().numpy().transpose(1,2,0).reshape(H, -1, 3) * 255. #NCHW->HNWC
                    input_img_resize = cv2.resize(input_img, (2*W, H))

                    img = rgb8
                    img = np.concatenate([input_img_resize, img], axis=0)
                    imageio.imwrite(filename, to8b(img/255.))

                    input_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}_input.png'.format(int(test_data['pose_index'][j]), view_id))
                    gt_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}_gt.png'.format(int(test_data['pose_index'][j]), view_id))
                    pred_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}.png'.format(int(test_data['pose_index'][j]), view_id))
                    imageio.imwrite(input_filename, to8b(input_img/255.))
                    imageio.imwrite(gt_filename, to8b(gt_img.cpu().numpy()))
                    imageio.imwrite(pred_filename, to8b(img_pred.cpu().numpy()))

                    # calculate loss
                    test_loss = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    psnr = round(mse2psnr(test_loss).item(), 3)
                    ssim, lpips = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), mask_at_box[j].shape[0], mask_at_box[j].shape[1])

                    print("[Test] ", "Source pose:", int(test_data['pose_index'][j]), " Target pose:", int(test_data['pose_index'][j]), " View:", view_id, " Loss:", round(test_loss.item(), 5), " PSNR:", {psnr}, " SSIM: ", {round(ssim, 3)}, " LPIPS: ", {round(lpips, 3)})
                    
                    psnr_sub_view.append(psnr)
                    ssim_sub_view.append(ssim)
                    lpips_sub_view.append(lpips)

            avg_psnr = np.array(psnr_sub_view).mean()
            np.save(savedir_human+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(avg_psnr))
            avg_ssim = np.array(ssim_sub_view).mean()
            np.save(savedir_human+'/ssim_{}.npy'.format(int(avg_ssim*100)), np.array(avg_ssim))
            avg_lpips = np.array(lpips_sub_view).mean()
            np.save(savedir_human+'/lpips_{}.npy'.format(int(avg_lpips*100)), np.array(avg_lpips))

            total_psnr.append(psnr_sub_view)
            total_ssim.append(ssim_sub_view)
            total_lpips.append(lpips_sub_view)

        # psnr
        avg_psnr = np.array(total_psnr).mean()
        np.save(savedir+'/novel_view/'+f'obs_view_{obs_view}'+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(total_psnr))
        # ssim
        avg_ssim = np.array(total_ssim).mean()
        np.save(savedir+'/novel_view/'+f'obs_view_{obs_view}'+'/ssim_{}.npy'.format(int(avg_ssim*100)), np.array(total_ssim))
        # lpips
        avg_lpips = np.array(total_lpips).mean()
        np.save(savedir+'/novel_view/'+f'obs_view_{obs_view}'+'/lpips_{}.npy'.format(int(avg_lpips*100)), np.array(total_lpips))

    # novel pose synthesis with obs image from the np_pose_start pose
    pose_start = np_pose_start # 2
    for obs_view in obs_view_lst:

        total_psnr = []
        total_ssim = []
        total_lpips = []

        for p, human_data_path in enumerate(all_humans):
            data_root = human_data_path.strip()
            human_name = os.path.basename(data_root)

            test_dataset_kwargs = dnnlib.EasyDict(class_name=class_name, data_root=data_root, split='test', multi_person=False, num_instance=1, poses_start=pose_start, poses_interval=pose_interval, poses_num=pose_num, image_scaling=image_scaling, white_back=white_back, sample_obs_view=sample_obs_view, fix_obs_view=fix_obs_view)
            test_set = dnnlib.util.construct_class_by_name(**test_dataset_kwargs) # subclass of training.dataset.Dataset
            test_set.obs_pose_index = pose_start
            test_set.obs_view_index = obs_view
            
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        
            psnr_sub_view = []
            ssim_sub_view = []
            lpips_sub_view = []

            savedir_human = os.path.join(savedir, 'novel_pose', f'obs_view_{obs_view}', human_name)
            os.makedirs(savedir_human, exist_ok=True)

            for k, test_data in enumerate(test_loader):

                view_id = k % test_set.camera_view_num

                if test_data['pose_index'][0] == np_pose_start or view_id % data_interval != 0:
                    continue

                print("novel pose subject: ", human_name, " obs_view: ", obs_view)

                test_data = to_cuda(device, test_data)

                gen_img = model(test_data, torch.randn(1, 512).to(device), torch.zeros((1, 25)).to(device), \
                    neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, test_flag=True)

                H, W = test_data['img_all'].shape[-2], test_data['img_all'].shape[-1]
                mask_at_box = test_data['mask_at_box_large_all'].reshape(test_data['mask_at_box_large_all'].shape[0], H, W)

                for j in range(batch_size):

                    img_pred = (gen_img['image'][j] / 2 + 0.5).permute(1,2,0)
                    real_img = test_data['img_all'][j]
                    gt_img = real_img[j].permute(1,2,0)

                    rgb8 = to8b(img_pred.cpu().numpy())
                    gt_rgb8 = to8b(gt_img.cpu().numpy())
                    rgb8 = np.concatenate([rgb8, gt_rgb8], axis=1)

                    filename = os.path.join(savedir_human, '{:02d}_{:02d}_{:02d}.png'.format(int(test_set.obs_pose_index), int(test_data['pose_index'][j]), view_id))

                    input_img = test_data['obs_img_all'][j,0].cpu().numpy().transpose(1,2,0).reshape(H, -1, 3) * 255. #NCHW->HNWC
                    # input_img_resize = cv2.resize(input_img, (H*2, int(input_img.shape[0] * W*2 / input_img.shape[1])))
                    input_img_resize = cv2.resize(input_img, (2*W, H))

                    img = rgb8
                    img = np.concatenate([input_img_resize, img], axis=0)
                    imageio.imwrite(filename, to8b(img/255.))

                    input_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}_input.png'.format(int(test_set.obs_pose_index), int(test_set.obs_view_index)))
                    gt_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}_gt.png'.format(int(test_data['pose_index'][j]), view_id))
                    pred_filename = os.path.join(savedir_human, 'frame{:04d}_view{:04d}.png'.format(int(test_data['pose_index'][j]), view_id))
                    imageio.imwrite(input_filename, to8b(input_img/255.))
                    imageio.imwrite(gt_filename, to8b(gt_img.cpu().numpy()))
                    imageio.imwrite(pred_filename, to8b(img_pred.cpu().numpy()))

                    # calculate loss
                    test_loss = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    psnr = round(mse2psnr(test_loss).item(), 3)

                    ssim, lpips = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), mask_at_box[j].shape[0], mask_at_box[j].shape[1])

                    print("[Test] ", "Source pose:", test_set.obs_pose_index, " Target pose:", int(test_data['pose_index'][j]), " View:", view_id, \
                        " Loss:", round(test_loss.item(), 5), " PSNR:", {psnr}, " SSIM: ", {round(ssim, 3)}, " LPIPS: ", {round(lpips, 3)})
                    
                    psnr_sub_view.append(psnr)
                    ssim_sub_view.append(ssim)
                    lpips_sub_view.append(lpips)

            avg_psnr = np.array(psnr_sub_view).mean()
            np.save(savedir_human+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(avg_psnr))
            
            avg_ssim = np.array(ssim_sub_view).mean()
            np.save(savedir_human+'/ssim_{}.npy'.format(int(avg_ssim*100)), np.array(avg_ssim))

            avg_lpips = np.array(lpips_sub_view).mean()
            np.save(savedir_human+'/lpips_{}.npy'.format(int(avg_lpips*100)), np.array(avg_lpips))

            total_psnr.append(psnr_sub_view)
            total_ssim.append(ssim_sub_view)
            total_lpips.append(lpips_sub_view)

        # psnr
        avg_psnr = np.array(total_psnr).mean()
        np.save(savedir+'/novel_pose/'+f'obs_view_{obs_view}'+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(total_psnr))
        # ssim
        avg_ssim = np.array(total_ssim).mean()
        np.save(savedir+'/novel_pose/'+f'obs_view_{obs_view}'+'/ssim_{}.npy'.format(int(avg_ssim*100)), np.array(total_ssim))
        # lpips
        avg_lpips = np.array(total_lpips).mean()
        np.save(savedir+'/novel_pose/'+f'obs_view_{obs_view}'+'/lpips_{}.npy'.format(int(avg_lpips*100)), np.array(total_lpips))

    return

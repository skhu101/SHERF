# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

import imageio
import cv2
from pytorch_msssim import ssim
import lpips

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, input_data, z, c, swapping_prob, neural_rendering_resolution, use_sr_module=True, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        input_img = input_data['obs_img_all'][:,0]
        ws = self.G.mapping(z, c_gen_conditioning, input_img=input_img, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        gen_output = self.G.synthesis(ws, input_data, c, neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, update_emas=update_emas, noise_mode='none')
        
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
 
        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, input_data, real_img, real_c, gen_z, gen_c, gain, cur_nimg, use_sr_module=True, recons_loss=True, rank=0):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        # swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None
        swapping_prob = 0

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.nneural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        del real_img
        real_img = input_data['img_all'][:,0]
        real_img_raw = real_img #filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        mask_at_box_raw = input_data['mask_at_box_all'][:,0].reshape(input_data['mask_at_box_all'].shape[0], real_img_raw.shape[-2], real_img_raw.shape[-1])

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        self.blur_raw_target = False
        blur_sigma = 0
        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw_blur = upfirdn2d.filter2d(real_img_raw, f / f.sum())
            else:
                real_img_raw_blur = real_img_raw
        else:
            real_img_raw_blur = real_img_raw

        real_img_blur = {'image': input_data['img_all'][:,0], 'image_raw': real_img_raw_blur}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):

                gen_img, _gen_ws = self.run_G(input_data, gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module)
                # compute loss
                img_loss_raw = img2mse( gen_img['image_raw'].permute(0,2,3,1)[mask_at_box_raw] / 2 + 0.5, real_img['image_raw'].permute(0,2,3,1)[mask_at_box_raw])
                acc_loss_raw = img2mse( gen_img['weights_image'].permute(0,2,3,1)[mask_at_box_raw], input_data['bkgd_msk_all'].reshape(*input_data['bkgd_msk_all'].shape[:2], real_img_raw.shape[-2], real_img_raw.shape[-1]).type(torch.int8).permute(0,2,3,1)[mask_at_box_raw])
                ssim_raw = 0
                lpips_raw = 0
                for i in range(mask_at_box_raw.shape[0]):
                    # crop the object region
                    x, y, w, h = cv2.boundingRect(mask_at_box_raw[i].cpu().numpy().astype(np.uint8))
                    img_pred = gen_img['image_raw'][i][:, y:y + h, x:x + w].unsqueeze(0) / 2 + 0.5
                    img_gt = real_img['image_raw'][i][:, y:y + h, x:x + w].unsqueeze(0)
                    ssim_raw += ssim(img_pred, img_gt, data_range=1, size_average=False)
                    lpips_raw += loss_fn_vgg(img_pred, img_gt).reshape(-1)

                loss_Gmain_Dgen = torch.zeros(1).to(real_img_raw.device)#torch.nn.functional.softplus(-gen_logits).mean()
                training_stats.report('Loss/G_Dgen/loss', loss_Gmain_Dgen)
                if recons_loss:
                    loss_Gmain = 100*img_loss_raw + 10*acc_loss_raw + (1-ssim_raw) + lpips_raw + 0*loss_Gmain_Dgen
                training_stats.report('Loss/img_loss_raw', img_loss_raw)
                training_stats.report('Loss/acc_loss_raw', acc_loss_raw)
                training_stats.report('Loss/ssim_raw', ssim_raw)
                training_stats.report('Loss/lpips_raw', lpips_raw)
                training_stats.report('Loss/img_D', loss_Gmain_Dgen)
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
        
            return loss_Gmain, img_loss_raw, acc_loss_raw, ssim_raw, lpips_raw, loss_Gmain_Dgen.mean()

        # # Density Regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
        #     import pdb; pdb.set_trace()
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     # input_img = input_data['obs_img_all'][:,0]
        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     # if self.style_mixing_prob > 0:
        #     #     with torch.autograd.profiler.record_function('style_mixing'):
        #     #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #     #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #     #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()
        
        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(input_data, gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img_blur['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth']) * 2 - 1
                real_img_tmp_image_raw = real_img_blur['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth']) * 2 - 1
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

            if phase in ['Dmain', 'Dboth']:
                return loss_Dgen.mean(), loss_Dreal.mean()
            elif phase in ['Dreg', 'Dboth']:
                return loss_Dr1.mean()


# #----------------------------------------------------------------------------

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from ipaddress import _IPAddressBase
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer, read_pickle, SMPL_to_tensor
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
import spconv.pytorch as spconv
from torchvision.models import resnet18
import os
import pickle
import numpy as np
import imageio

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        use_1d_feature,
        use_2d_feature,
        use_3d_feature, 
        use_trans,
        use_NeRF_decoder,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer(use_1d_feature=use_1d_feature, use_2d_feature=use_2d_feature, use_3d_feature=use_3d_feature, use_trans=use_trans, use_NeRF_decoder=use_NeRF_decoder)
        self.ray_sampler = RaySampler()
        self.encoder_2d = ResNet18Classifier()
        self.encoder_2d_feature = ResNet18Classifier()
        self.conv1d_projection = nn.Conv1d(96, 32, 1)
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        if not use_NeRF_decoder:
            self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 3})
        else:
            self.decoder = NeRFDecoder(32)
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.use_1d_feature = use_1d_feature
        self.use_2d_feature = use_2d_feature
        self.use_3d_feature = use_3d_feature

        self._last_planes = None

    def mapping(self, z, c, input_img=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # del z
        z = self.encoder_2d(input_img)
        # z: [1, 512]; c: [1, 25]
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, input_data, c, neural_rendering_resolution=None, use_sr_module=True, update_emas=False, cache_backbone=False, use_cached_backbone=False, test_flag=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        ray_origins, ray_directions = input_data['ray_o_all'][:,0], input_data['ray_d_all'][:,0]
        near = input_data['near_all'][:,0]
        far = input_data['far_all'][:,0]

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs) #[1, 96, 256, 256]
        if cache_backbone:
            self._last_planes = planes

        if self.use_3d_feature:
            # get vertex feature to form sparse convolution tensor
            obs_input_img = input_data['obs_img_all'][:,0] # [bs, 3, 512, 512]
            obs_input_feature = self.encoder_2d_feature(obs_input_img, extract_feature=True) # [bs, 64, 256, 256]

            bs = obs_input_img.shape[0]
            obs_vertex_pts = input_data['obs_vertices'] # [bs, 6890, 3]
            obs_uv, obs_smpl_vertex_mask = self.renderer.projection(obs_vertex_pts.reshape(bs, -1, 3), input_data['obs_R_all'], input_data['obs_T_all'], input_data['obs_K_all'], self.renderer.SMPL_NEUTRAL['f']) # [bs, N, 6890, 3]
            obs_uv = obs_uv.view(-1, *obs_uv.shape[2:]) # [bs, N_rays*N_rand, 2]
            obs_uv_ = 2.0 * obs_uv.unsqueeze(2).type(torch.float32) / torch.Tensor([obs_input_img.shape[-1], obs_input_img.shape[-2]]).to(obs_input_img.device) - 1.0
            obs_vertex_feature = F.grid_sample(obs_input_feature, obs_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]

            # obs_img = obs_input_img.reshape(-1, *obs_input_img.shape[2:])
            obs_vertex_rgb = F.grid_sample(obs_input_img, obs_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]

            # obs_vertex_rgb = obs_vertex_rgb.view(bs, -1 , *obs_vertex_rgb.shape[1:]).transpose(2,3)
            sh = obs_vertex_rgb.shape
            obs_vertex_rgb = self.renderer.rgb_enc(obs_vertex_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 
            obs_vertex_3d_feature = torch.cat((obs_vertex_feature, obs_vertex_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 
            obs_vertex_3d_feature = self.conv1d_projection(obs_vertex_3d_feature.permute(0,2,1)).permute(0,2,1)

            obs_vertex_3d_feature[obs_smpl_vertex_mask==0] = 0

            ## vertex points in SMPL coordinates
            smpl_obs_pts = torch.matmul(obs_vertex_pts.reshape(bs, -1, 3) - input_data['obs_params']['Th'], input_data['obs_params']['R'])

            ## coarse deform target to caonical
            coarse_obs_vertex_canonical_pts = self.renderer.coarse_deform_target2c(input_data['obs_params'], input_data['obs_vertices'], input_data['t_params'], smpl_obs_pts) # [bs, N_rays*N_rand, 3]       

            # prepare sp input
            obs_sp_input, _ = self.prepare_sp_input(input_data['t_vertices'], coarse_obs_vertex_canonical_pts)

            canonical_sp_conv_volume = spconv.core.SparseConvTensor(obs_vertex_3d_feature.reshape(-1, obs_vertex_3d_feature.shape[-1]), obs_sp_input['coord'], obs_sp_input['out_sh'], obs_sp_input['batch_size']) # [bs, 32, 96, 320, 384] z, y, x

        else:
            bs = input_data['obs_img_all'].shape[0]
            canonical_sp_conv_volume = None
            obs_sp_input = None
            obs_input_img = input_data['obs_img_all'][:,0] # [bs, 3, 512, 512]
            obs_input_feature = self.encoder_2d_feature(obs_input_img, extract_feature=True) # [bs, 64, 256, 256]
            obs_vertex_pts = input_data['obs_vertices'] # [bs, 6890, 3]
            _, obs_smpl_vertex_mask = self.renderer.projection(obs_vertex_pts.reshape(bs, -1, 3), input_data['obs_R_all'], input_data['obs_T_all'], input_data['obs_K_all'], self.renderer.SMPL_NEUTRAL['f']) # [bs, N, 6890, 3]


        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # [1, 3, 32, 256, 256]

        if test_flag:
            self.rendering_kwargs.update({'density_noise': 0})

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, obs_input_img, obs_input_feature, canonical_sp_conv_volume, obs_smpl_vertex_mask, obs_sp_input, self.decoder, ray_origins, 
            ray_directions, near, far, input_data, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H, W = input_data['obs_img_all'].shape[-2:] #self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        if use_sr_module:
            sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'weights_image': weights_image}

    def prepare_sp_input(self, vertex, xyz):

        self.big_box = True
        # obtain the bounds for coord construction
        min_xyz = torch.min(vertex, dim=1)[0]
        max_xyz = torch.max(vertex, dim=1)[0]

        if self.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[:, 2] -= 0.05
            max_xyz[:, 2] += 0.05

        bounds = torch.cat([min_xyz.unsqueeze(1), max_xyz.unsqueeze(1)], axis=1)


        dhw = xyz[:, :, [2, 1, 0]]
        min_dhw = min_xyz[:, [2, 1, 0]]
        max_dhw = max_xyz[:, [2, 1, 0]]
        voxel_size = torch.Tensor([0.005, 0.005, 0.005]).to(device=dhw.device)
        coord = torch.round((dhw - min_dhw.unsqueeze(1)) / voxel_size).to(dtype=torch.int32)

        # construct the output shape
        out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).to(dtype=torch.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x 
        sh = dhw.shape # torch.Size([1, 6890, 3])
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(coord)
        coord = coord.view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(out_sh, dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['bounds'] = bounds

        return sp_input, _#, pc_features

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, input_data, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, use_sr_module=True, update_emas=False, cache_backbone=False, use_cached_backbone=False, test_flag=False, **synthesis_kwargs):
        # Render a batch of generated images.
        input_img = input_data['obs_img_all'][:,0]
        ws = self.mapping(z, c, input_img=input_img, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, input_data, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, test_flag=test_flag, **synthesis_kwargs)

#----------------------------------------------------------------------------

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class NeRFDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        W = 128
        self.with_viewdirs = True
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = n_features + 39
        # self.feature_projection = nn.Linear(96*3, 96)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        nerf_input_ch_2 = n_features + W # 96 fused feature + 256 alpha feature
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        
    def forward(self, ray_points, sampled_features, ray_directions):

        # sampled_features[:] = 0
        # point_2d_feature = self.feature_projection(sampled_features.permute(1,0,2).reshape(ray_points.shape[0], -1))

        point_2d_feature_1 = sampled_features[0]
        point_2d_feature_2 = sampled_features[1]

        x = ray_points
        x = torch.cat((x, point_2d_feature_1), dim=-1)
        h = x

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        sigma = self.alpha_linear(h)
        feature = self.feature_linear(h)

        if self.with_viewdirs:
            h = torch.cat([feature, ray_directions, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001

        return {'rgb': rgb.unsqueeze(0), 'sigma': sigma.unsqueeze(0)}

#----------------------------------------------------------------------------

class ResNet18Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18Classifier, self).__init__()
        self.backbone = resnet18(pretrained=True)

    def forward(self, x, extract_feature=False):
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if not extract_feature:
            x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if extract_feature:
            return x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

#----------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

from ipaddress import _IPAddressBase
import math
import os
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import pickle
import numpy as np
from pytorch3d.ops.knn import knn_points
import spconv.pytorch as spconv
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
import torch.autograd.profiler as profiler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = torch.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps 
    arr[..., 0] /= lens
    arr[..., 1] /= lens
    arr[..., 2] /= lens
    return arr 

def compute_normal(vertices, faces):
    # norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    norm = torch.zeros(vertices.shape, dtype=vertices.dtype).cuda()
    tris = vertices[:, faces] # [bs, 13776, 3, 3]
    # n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n = torch.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]) 
    normalize_v3(n)
    norm[:, faces[:, 0]] += n
    norm[:, faces[:, 1]] += n
    norm[:, faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs = joints.shape[0]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, 24, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, 24, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

# @profile
def get_transform_params_torch(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']

    v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # bs x 24 x 3 x 3
    rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]

    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    # return torch.tensor([[[1, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 1]],
    #                         [[1, 0, 0],
    #                         [0, 0, 1],
    #                         [0, 1, 0]],
    #                         [[0, 0, 1],
    #                         [1, 0, 0],
    #                         [0, 1, 0]]], dtype=torch.float32)
    # fix for xy, xz, zy projections
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    coordinates = 2 * (coordinates - box_warp[:, :1]) / (box_warp[:, 1:2] - box_warp[:, :1]) - 1 # TODO: add specific box bounds
    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1) # [3, 1, 786432, 2]
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self, use_1d_feature=True, use_2d_feature=True, use_3d_feature=True, use_trans=False, use_NeRF_decoder=False):
        super().__init__()
        self.use_1d_feature = use_1d_feature
        self.use_2d_feature = use_2d_feature
        self.use_3d_feature = use_3d_feature
        self.use_trans = use_trans
        self.use_NeRF_decoder = use_NeRF_decoder
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        self.encoder_3d = SparseConvNet(num_layers=4)
        self.conv1d_projection = nn.Conv1d(192, 96, 1)
        if use_1d_feature and use_2d_feature and use_3d_feature:
            self.conv1d_reprojection = nn.Conv1d(96, 32, 1)
        elif (use_1d_feature and use_2d_feature) or (use_1d_feature and use_3d_feature) or (use_3d_feature and use_2d_feature):
            self.conv1d_reprojection = nn.Conv1d(64, 32, 1)
        self.transformer = None if not use_trans else Transformer(32)
        self.rgb_enc = PositionalEncoding(num_freqs=5)
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
        # self.view_enc = PositionalEncoding(num_freqs=5)

        # load SMPL model
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=torch.device('cuda', torch.cuda.current_device()))

    def forward(self, planes, obs_input_img, obs_input_feature, canonical_sp_conv_volume, obs_smpl_vertex_mask, obs_sp_input, decoder, ray_origins, ray_directions, near, far, input_data, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        
        # if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        #     ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        #     is_ray_valid = ray_end > ray_start
        #     if torch.any(is_ray_valid).item():
        #         ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
        #         ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        #     depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        # else:
        #     # Create stratified depth samples
        #     depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        depths_coarse = self.sample_stratified(ray_origins, near, far, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        R = input_data['params']['R'] # [bs, 3, 3]
        Th = input_data['params']['Th'] #.astype(np.float32) [bs, 1, 3]
        smpl_query_pts = torch.matmul(sample_coordinates - Th, R) # [bs, N_rays*N_samples, 3]
        smpl_query_viewdir = torch.matmul(sample_directions, R)

        # human sample
        tar_smpl_pts = input_data['vertices'] # [bs, 6890, 3]
        tar_smpl_pts = torch.matmul(tar_smpl_pts - Th, R) # [bs, 6890, 3]
        distance, vertex_id, _ = knn_points(smpl_query_pts.float(), tar_smpl_pts.float(), K=1)
        distance = distance.view(distance.shape[0], -1)
        pts_mask = torch.zeros_like(smpl_query_pts[...,0]).int()
        threshold = 0.05 ** 2 
        pts_mask[distance < threshold] = 1
        smpl_query_pts = smpl_query_pts[pts_mask==1].unsqueeze(0)
        smpl_query_viewdir = smpl_query_viewdir[pts_mask==1].unsqueeze(0)   

        coarse_canonical_pts, coarse_canonical_viewdir = self.coarse_deform_target2c(input_data['params'], input_data['vertices'], input_data['t_params'], smpl_query_pts, smpl_query_viewdir)

        if self.use_2d_feature:
            # extract pixel aligned 2d feature
            bs = coarse_canonical_pts.shape[0]
            _, world_src_pts, _ = self.coarse_deform_c2source(input_data['obs_params'], input_data['t_params'], input_data['t_vertices'], coarse_canonical_pts)

            src_uv = self.projection(world_src_pts.reshape(bs, -1, 3), input_data['obs_R_all'], input_data['obs_T_all'], input_data['obs_K_all']) # [bs, N, 6890, 3]
            src_uv = src_uv.view(-1, *src_uv.shape[2:])
            src_uv_ = 2.0 * src_uv.unsqueeze(2).type(torch.float32) / torch.Tensor([obs_input_img.shape[-1], obs_input_img.shape[-2]]).to(obs_input_img.device) - 1.0
            point_pixel_feature = F.grid_sample(obs_input_feature, src_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]
    
            # extract pixel aligned rgb feature
            point_pixel_rgb = F.grid_sample(obs_input_img, src_uv_, align_corners=True)[..., 0].permute(0,2,1)  # [B, N, C]
            
            sh = point_pixel_rgb.shape
            point_pixel_rgb = self.rgb_enc(point_pixel_rgb.reshape(-1,3)).reshape(*sh[:2], 33)[..., :32] # [bs, N_rays*N_samples, 32] 
            point_2d_feature = torch.cat((point_pixel_feature, point_pixel_rgb), dim=-1) # [bs, N_rays*N_samples, 96] 

        else:
            point_2d_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)

        if self.use_3d_feature:
            grid_coords = self.get_grid_coords(coarse_canonical_pts, obs_sp_input)
            # grid_coords = grid_coords.view(bs, -1, 3)
            grid_coords = grid_coords[:, None, None]
            point_3d_feature = self.encoder_3d(canonical_sp_conv_volume, grid_coords) # torch.Size([b, 390, 1024*64])
            point_3d_feature = self.conv1d_projection(point_3d_feature.permute(0,2,1)).permute(0,2,1) # torch.Size([b, N, 96])
        else:
            point_3d_feature = torch.zeros((*smpl_query_pts.shape[:2], 96)).to(smpl_query_pts.device)

        out = {}
        chunk = 700000
        for i in range(0, coarse_canonical_pts.shape[1], chunk):
            out_part = self.run_model(planes, point_2d_feature[:, i:i+chunk], point_3d_feature[:, i:i+chunk], decoder, coarse_canonical_pts[:, i:i+chunk], coarse_canonical_viewdir[:, i:i+chunk], input_data['t_world_bounds'], pts_mask, rendering_options)
            for k in out_part.keys():
                if k not in out.keys():
                    out[k] = []
                out[k].append(out_part[k]) 
        out = {k : torch.cat(out[k], 1) for k in out.keys()}

        colors_coarse = torch.zeros((*pts_mask.shape, 3), device=pts_mask.device)
        densities_coarse = torch.zeros((*pts_mask.shape, 1), device=pts_mask.device)
        colors_coarse[pts_mask==1] = out['rgb'].squeeze(0)
        densities_coarse[pts_mask==1] = out['sigma'].squeeze(0)
        densities_coarse[pts_mask==0] = -80

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, ray_directions, rendering_options)


        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, point_2d_feature, point_3d_feature, decoder, sample_coordinates, sample_directions, box_warp, pts_mask, options):
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=box_warp)

        # if point_2d_feature is not None and point_3d_feature is not None:
        if self.use_1d_feature and not self.use_2d_feature and not self.use_3d_feature:
            sampled_features = sampled_features
        elif self.use_1d_feature and self.use_2d_feature and not self.use_3d_feature:
            sampled_features_combine = torch.cat([sampled_features, point_2d_feature.reshape(*point_2d_feature.shape[:2], 3, 32).permute(0, 2,1,3)], dim=-1)
            sampled_features = self.conv1d_reprojection(sampled_features_combine.reshape(-1, *sampled_features_combine.shape[-2:]).permute(0,2,1)).permute(0,2,1).reshape(-1, 3, *sampled_features.shape[-2:])
        elif self.use_1d_feature and self.use_3d_feature and not self.use_2d_feature:
            sampled_features_combine = torch.cat([sampled_features, point_3d_feature.reshape(*point_3d_feature.shape[:2], 3, 32).permute(0, 2,1,3)], dim=-1)
            sampled_features = self.conv1d_reprojection(sampled_features_combine.reshape(-1, *sampled_features_combine.shape[-2:]).permute(0,2,1)).permute(0,2,1).reshape(-1, 3, *sampled_features.shape[-2:])
        elif self.use_2d_feature and self.use_3d_feature and not self.use_1d_feature:
            sampled_features_combine = torch.cat([point_2d_feature.reshape(*point_2d_feature.shape[:2], 3, 32).permute(0, 2,1,3), point_3d_feature.reshape(*point_2d_feature.shape[:2], 3, 32).permute(0, 2,1,3)], dim=-1)
            sampled_features = self.conv1d_reprojection(sampled_features_combine.reshape(-1, *sampled_features_combine.shape[-2:]).permute(0,2,1)).permute(0,2,1).reshape(-1, 3, *sampled_features.shape[-2:])
        elif self.use_1d_feature and self.use_2d_feature and self.use_3d_feature:
            # if not self.use_NeRF_decoder:
            #     sampled_features_combine = torch.cat([sampled_features, point_2d_feature.reshape(*point_2d_feature.shape[:2], 3, 32).permute(0, 2,1,3), point_3d_feature.reshape(*point_3d_feature.shape[:2], 3, 32).permute(0, 2,1,3)], dim=-1)
            #     sampled_features = self.conv1d_reprojection(sampled_features_combine.reshape(-1, *sampled_features_combine.shape[-2:]).permute(0,2,1)).permute(0,2,1).reshape(-1, 3, *sampled_features.shape[-2:])
            # else:
            #     sampled_features_combine = torch.cat([sampled_features.permute(0, 1, 3, 2).reshape(point_2d_feature.shape[0], -1, point_2d_feature.shape[1]), point_2d_feature.permute(0, 2, 1), point_3d_feature.permute(0, 2, 1)], dim=0)
            #     sampled_features = self.conv1d_reprojection(sampled_features_combine)[None].permute(0, 1, 3, 2)
            sampled_features_combine = torch.cat([sampled_features, point_2d_feature.reshape(*point_2d_feature.shape[:2], 3, 32).permute(0, 2,1,3), point_3d_feature.reshape(*point_3d_feature.shape[:2], 3, 32).permute(0, 2, 1, 3)], dim=-1)
            sampled_features = self.conv1d_reprojection(sampled_features_combine.reshape(-1, *sampled_features_combine.shape[-2:]).permute(0,2,1)).permute(0,2,1).reshape(-1, 3, *sampled_features.shape[-2:])

        if self.use_trans:
            sampled_features = self.transformer(sampled_features.permute(0,2,1,3).reshape(-1, sampled_features.shape[1], sampled_features.shape[-1])).permute(1,0,2).reshape(-1, 3, *sampled_features.shape[2:])

        if not self.use_NeRF_decoder:
            out = decoder(sampled_features, sample_directions)
        else:
            out = decoder(self.pos_enc(sample_coordinates.squeeze(0)), sampled_features.squeeze(0), self.view_enc(sample_directions.squeeze(0)))

        # out = decoder(sampled_features.permute(0,2,1,3)[pts_mask==1].permute(1,0,2).unsqueeze(0), sample_directions[pts_mask==1].unsqueeze(0))
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

    def get_grid_coords(self, pts, sp_input):

        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def coarse_deform_target2c(self, params, vertices, t_params, query_pts, query_viewdirs = None):

        bs = query_pts.shape[0]
        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        smpl_pts = torch.matmul((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], 24)#.to(vertices.device)
        # From smpl space target pose to smpl space canonical pose
        A = torch.matmul(bweights, A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        can_pts = torch.matmul(R_inv, can_pts[..., None]).squeeze(-1)

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(R_inv, query_viewdirs[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'] #.to(vertices.device).float()
            pose_ = params['poses']
            ident = torch.eye(3).to(pose_.device).float()
            batch_size = pose_.shape[0] #1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - pose_offsets

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs']  #.to(pose_.device)
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'], (batch_size, 1, 10, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - shape_offset

        # From T To Big Pose        
        big_pose_params = t_params #self.big_pose_params(params)

        if self.mean_shape:
            pose_ = big_pose_params['poses']
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts + pose_offsets

            # To mean shape
            # shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(big_pose_params['shapes'].cuda(), (batch_size, 1, 10, 1))).squeeze(-1)
            # shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            # can_pts = can_pts + shape_offset

        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], can_pts[..., None]).squeeze(-1)
        can_pts = can_pts + A[..., :3, 3]

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(A[..., :3, :3], query_viewdirs[..., None]).squeeze(-1)
            return can_pts, query_viewdirs

        return can_pts

    def coarse_deform_c2source(self, params, t_params, t_vertices, query_pts, weights_correction=0):
        bs = query_pts.shape[0]
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], 24).cuda()

        # add weights_correction, normalize weights
        bweights = bweights + 0.2 * weights_correction # torch.Size([30786, 24])
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = t_params
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offsets

            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, 10, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offsets

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, 24, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def projection(self, query_pts, R, T, K, face=None):
        RT = torch.cat([R, T], -1)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=1), repeats=RT.shape[1], dim=1) #[bs, view_num, , 3]
        xyz = torch.matmul(RT[:, :, None, :, :3].float(), xyz[..., None].float()) + RT[:, :, None, :, 3:].float()

        if face is not None:
            # compute the normal vector for each vertex
            smpl_vertex_normal = compute_normal(query_pts, face) # [bs, 6890, 3]
            smpl_vertex_normal_cam = torch.matmul(RT[:, :, None, :, :3].float(), smpl_vertex_normal[:, None, :, :, None].float()) # [bs, 1, 6890, 3, 1]
            smpl_vertex_mask = (smpl_vertex_normal_cam * xyz).sum(-2).squeeze(1).squeeze(-1) < 0 

        # xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xyz = torch.matmul(K[:, :, None].float(), xyz)[..., 0]
        xy = xyz[..., :2] / (xyz[..., 2:] + 1e-5)

        if face is not None:
            return xy, smpl_vertex_mask 
        else:
            return xy

#----------------------------------------------------------------------------

class SparseConvNet(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, num_layers=2):
        super(SparseConvNet, self).__init__()
        self.num_layers = num_layers

        # self.conv0 = double_conv(3, 16, 'subm0')
        # self.down0 = stride_conv(16, 32, 'down0')

        # self.conv1 = double_conv(32, 32, 'subm1')
        # self.down1 = stride_conv(32, 64, 'down1')

        # self.conv2 = triple_conv(64, 64, 'subm2')
        # self.down2 = stride_conv(64, 128, 'down2')

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 96, 'down2')

        self.conv3 = triple_conv(96, 96, 'subm3')
        self.down3 = stride_conv(96, 96, 'down3')

        self.conv4 = triple_conv(96, 96, 'subm4')

        self.channel = 32

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        features = []

        net = self.conv0(x)
        net = self.down0(net)

        # point_normalied_coords = point_normalied_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if self.num_layers > 1:
            net = self.conv1(net)
            net1 = net.dense()
            # torch.Size([1, 32, 1, 1, 4096])
            feature_1 = F.grid_sample(net1, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_1)
            self.channel = 32
            net = self.down1(net)
        
        if self.num_layers > 2:
            net = self.conv2(net)
            net2 = net.dense()
            # torch.Size([1, 64, 1, 1, 4096])
            feature_2 = F.grid_sample(net2, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_2)
            self.channel = 64
            net = self.down2(net)
        
        if self.num_layers > 3:
            net = self.conv3(net)
            net3 = net.dense()
            # 128
            feature_3 = F.grid_sample(net3, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_3)
            self.channel = 128
            net = self.down3(net)
        
        if self.num_layers > 4:
            net = self.conv4(net)
            net4 = net.dense()
            # 256
            feature_4 = F.grid_sample(net4, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_4)

        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4)).transpose(1,2)

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    # tmp = spconv.SubMConv3d(in_channels,
    #                       out_channels,
    #                       3,
    #                       bias=False,
    #                       indice_key=indice_key)
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())

#----------------------------------------------------------------------------

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=None, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        # self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.freqs = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = torch.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            if x.shape[0]==0:
                embed = embed.view(x.shape[0], self.num_freqs*6)
            else:
                embed = embed.view(x.shape[0], -1)
                
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

#----------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) # torch.Size([30786, 3, 768])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim=32, depth=1, heads=3, dim_head=16, mlp_dim=32, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

#----------------------------------------------------------------------------
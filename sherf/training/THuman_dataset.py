from torch.utils.data import DataLoader, dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import time
import copy
from random import sample
from smpl.smpl_numpy import SMPL

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    # TODO
    # mask_at_box = p_mask_at_box.sum(-1) >= 1

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_THuman_batch(img, msk, K, R, T, bounds, image_scaling, white_back=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if mask_bkgd:
        # img[bound_mask != 1] = 1 if white_back else 0
        img[bound_mask != 1] = 0 

    # rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    coord = np.zeros([len(ray_o), 2]).astype(np.int64)
    bkgd_msk = msk

    return img, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk


class THumanDatasetBatch(Dataset):
    def __init__(self, data_root=None, split='test', multi_person=False, num_instance=1, poses_start=0, poses_interval=1, poses_num=20, image_scaling=1, white_back=False, sample_obs_view=False, fix_obs_view=True, resolution=None):
        super(THumanDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_scaling = image_scaling
        self.white_back = white_back
        self.sample_obs_view = sample_obs_view
        self.fix_obs_view = fix_obs_view
        self.camera_view_num = 24

        self.output_view = [x for x in range(self.camera_view_num)]

        self.poses_start = poses_start # start index 0
        self.poses_interval = poses_interval # interval 1
        self.poses_num = poses_num # number of used poses

        # observation pose and view
        self.obs_pose_index = None
        self.obs_view_index = None

        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.output_view]
            for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
        ]) # view index of output_view, shape (num of poses, num of output_views)

        self.multi_person = multi_person

        self.num_instance = num_instance
        if self.multi_person:
            humans_data_root = os.path.dirname(data_root)
            self.humans_list = os.path.join(humans_data_root, 'human_list.txt')
            with open(self.humans_list) as f:
                humans_name = f.readlines()[0:num_instance]
        
        self.all_humans = [data_root] if not multi_person else [os.path.join(humans_data_root, x.strip()) for x in humans_name]
        print('num of subjects: ', len(self.all_humans))

        self.cams_all = []
        self.ims_all = []
        for subject_root in self.all_humans:
            ann_file = os.path.join(subject_root, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
            self.cams_all.append(annots['cams'])
            ims = np.array([
                np.array(ims_data['ims'])[self.output_view]
                for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])
            self.ims_all.append(ims)

        # prepare t pose and vertex
        smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
        self.big_pose_params = self.big_pose_params()
        t_vertices, _ = smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[pose_index][view_index].replace('\\', '/').replace('jpg', 'png'))
        msk = imageio.imread(msk_path)
        msk[msk!=0]=255
        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params_neutral", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()

        return world_bounds, vertices, params

    def big_pose_params(self):

        big_pose_params = {}
        # big_pose_params = copy.deepcopy(params)
        big_pose_params['R'] = np.eye(3).astype(np.float32)
        big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
        big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
        big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)

        return big_pose_params

    def __getitem__(self, index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """

        instance_idx = index // (self.poses_num * self.camera_view_num) if self.multi_person else 0
        pose_index = ( index % (self.poses_num * self.camera_view_num) ) // self.camera_view_num 
        #* self.poses_interval + self.poses_start
        view_index = index % self.camera_view_num

        self.data_root = self.all_humans[instance_idx]
        self.ims = self.ims_all[instance_idx]
        self.cams = self.cams_all[instance_idx]

        if pose_index >= len(self.ims):
            pose_index = np.random.randint(len(self.ims))

        img_all, ray_o_all, ray_d_all, near_all, far_all = [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, mask_at_box_large_all = [], [], []
        obs_img_all, obs_K_all, obs_R_all, obs_T_all = [], [], [], []
            
        # Load image, mask, K, D, R, T
        img_path = os.path.join(self.data_root, self.ims[pose_index][view_index].replace('\\', '/'))
        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk = np.array(self.get_mask(pose_index, view_index)) / 255.
        img[msk == 0] = 1 if self.white_back else 0

        K = np.array(self.cams['K'][view_index])
        D = np.array(self.cams['D'][view_index])
        R = np.array(self.cams['R'][view_index])
        T = np.array(self.cams['T'][view_index])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        # rescaling
        if self.image_scaling != 1:
            H, W = img.shape[:2]
            H, W = int(H * self.image_scaling), int(W * self.image_scaling)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2]*self.image_scaling

        # Prepare the smpl input, including the current pose and canonical pose
        # i: the pose index of all poses this person has, not the pose index of getitem input
        i = int(os.path.basename(img_path)[:-4])
        world_bounds, vertices, params = self.prepare_input(i)
        params = {'poses': np.squeeze(np.expand_dims(params['poses'].astype(np.float32), axis=0), axis=-1),
            'R': params['R'].astype(np.float32),
            'Th': params['Th'].astype(np.float32),
            'shapes': np.expand_dims(params['shapes'].astype(np.float32), axis=0)}

        # Sample rays in target space world coordinate
        img, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk = sample_ray_THuman_batch(
                img, msk, K, R, T, world_bounds, 1.0)

        mask_at_box_large = mask_at_box

        img = np.transpose(img, (2,0,1))

        # target view
        img_all.append(img)
        ray_o_all.append(ray_o)
        ray_d_all.append(ray_d)
        near_all.append(near)
        far_all.append(far)
        mask_at_box_all.append(mask_at_box)
        bkgd_msk_all.append(bkgd_msk)
        mask_at_box_large_all.append(mask_at_box_large)

        # training obs view data preparation
        if self.split == 'train':
            if self.sample_obs_view:
                self.obs_view_index = np.random.randint(self.camera_view_num)
            elif self.fix_obs_view:
                self.obs_view_index = 12

        if self.obs_pose_index is not None:
            obs_pose_index = int(self.obs_pose_index)
        else:
            obs_pose_index = pose_index

        # Load image, mask, K, D, R, T in observation space
        obs_img_path = os.path.join(self.data_root, self.ims[obs_pose_index][self.obs_view_index].replace('\\', '/'))
        obs_img = np.array(imageio.imread(obs_img_path).astype(np.float32) / 255.)
        obs_msk = np.array(self.get_mask(obs_pose_index, self.obs_view_index)) / 255.
        obs_img[obs_msk == 0] = 1 if self.white_back else 0

        obs_K = np.array(self.cams['K'][self.obs_view_index])
        obs_D = np.array(self.cams['D'][self.obs_view_index])
        obs_R = np.array(self.cams['R'][self.obs_view_index])
        obs_T = np.array(self.cams['T'][self.obs_view_index])
        obs_img = cv2.undistort(obs_img, obs_K, obs_D)
        obs_msk = cv2.undistort(obs_msk, obs_K, obs_D)

        # rescaling
        if self.image_scaling != 1:
            obs_img = cv2.resize(obs_img, (W, H), interpolation=cv2.INTER_AREA)
            obs_msk = cv2.resize(obs_msk, (W, H), interpolation=cv2.INTER_NEAREST)
            obs_K[:2] = obs_K[:2]*self.image_scaling

        obs_img = np.transpose(obs_img, (2,0,1))

        # Prepare smpl in the observation space
        # i: the pose index of all poses this person has, not the pose index of getitem input
        obs_i = int(os.path.basename(obs_img_path)[:-4])
        _, obs_vertices, obs_params = self.prepare_input(obs_i)
        obs_params = {'poses': np.squeeze(np.expand_dims(obs_params['poses'].astype(np.float32), axis=0), axis=-1),
            'R': obs_params['R'].astype(np.float32),
            'Th': obs_params['Th'].astype(np.float32),
            'shapes': np.expand_dims(obs_params['shapes'].astype(np.float32), axis=0)}

        # obs view
        obs_img_all.append(obs_img)
        obs_K_all.append(obs_K)
        obs_R_all.append(obs_R)
        obs_T_all.append(obs_T)

        # target view
        img_all = np.stack(img_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        mask_at_box_large_all = np.stack(mask_at_box_large_all, axis=0)

        # obs view 
        obs_img_all = np.stack(obs_img_all, axis=0)
        obs_K_all = np.stack(obs_K_all, axis=0)
        obs_R_all = np.stack(obs_R_all, axis=0)
        obs_T_all = np.stack(obs_T_all, axis=0)

        ret = {
            "instance_idx": instance_idx, # person instance idx
            'pose_index': pose_index * self.poses_interval + self.poses_start, # pose_index in selected poses

            # canonical space
            't_params': self.big_pose_params,
            't_vertices': self.t_vertices,
            't_world_bounds': self.t_world_bounds,

            # target view
            "params": params, # smpl params including smpl global R, Th
            'vertices': vertices, # world vertices
            'img_all': img_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all,
            'mask_at_box_large_all': mask_at_box_large_all,

            # obs view
            'obs_params': obs_params,
            'obs_vertices': obs_vertices,
            'obs_img_all': obs_img_all,
            'obs_K_all': obs_K_all,
            'obs_R_all': obs_R_all,
            'obs_T_all': obs_T_all,

        }

        return ret

    def __len__(self):
        return self.num_instance * self.poses_num * self.camera_view_num

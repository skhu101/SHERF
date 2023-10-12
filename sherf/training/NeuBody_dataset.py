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

def sample_ray_neubody_batch(img, msk, K, R, T, bounds, image_scaling, white_back=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    # img_ray_d = ray_d.copy()
    # img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if mask_bkgd:
        img[bound_mask != 1] = 0 #1 if white_back else 0

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


class NeuBodyDatasetBatch(Dataset):
    def __init__(self, data_root, split='test', border=5, N_rand=1024*32, multi_person=False, num_instance=1, 
                poses_start=0, poses_interval=5, poses_num=100, image_scaling=0.5, white_back=False, sample_obs_view=False, fix_obs_view=False, resolution=None):
        super(NeuBodyDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        self.sample_obs_view = sample_obs_view
        self.fix_obs_view = fix_obs_view
        self.camera_view_num = 20

        self.output_view = [x for x in range(self.camera_view_num)]

        self.poses_start = poses_start # start index 0
        self.poses_interval = poses_interval # interval 1
        self.poses_num = poses_num # number of used poses 30

        # observation pose and view
        self.obs_pose_index = None
        self.obs_view_index = None

        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.output_view]
            for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
        ])

        if 'CoreView_313' in data_root or 'CoreView_315' in data_root:
            for i in range(self.ims.shape[0]):
                self.ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in self.ims[i]]

        self.nrays = N_rand
        self.border = border
        self.image_scaling = image_scaling
        self.num_instance = num_instance

        self.multi_person = multi_person
        humans_data_root = os.path.join(os.path.dirname(data_root))
        humans_name = [
            "CoreView_386", "CoreView_387", "CoreView_390",
            "CoreView_392", "CoreView_393", "CoreView_394"
        ] # "CoreView_377", CoreView_313, CoreView_315, "CoreView_396" 
        self.all_humans = [data_root] if not multi_person else [os.path.join(humans_data_root, x.strip()) for x in humans_name]
        print(self.all_humans)

        self.ims_all = []
        self.cams_all = []
        self.cam_inds_all = []
        for subject_root in self.all_humans:
            ann_file = os.path.join(subject_root, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
            cams = annots['cams']
            ims = np.array([
                np.array(ims_data['ims'])[self.output_view]
                for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])

            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[self.output_view]
                for ims_data in annots['ims'][self.poses_start:self.poses_start + self.poses_num * self.poses_interval][::self.poses_interval]
            ])

            if 'CoreView_313' in subject_root or 'CoreView_315' in subject_root:
                for i in range(ims.shape[0]):
                    ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in ims[i]]

            self.ims_all.append(ims)
            self.cams_all.append(cams)

            self.cam_inds_all.append(cam_inds)

        # prepare t pose and vertex
        smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
        self.big_pose_params = self.prepare_big_pose_params()
        t_vertices, _ = smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    def prepare_big_pose_params(self):

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

    def get_mask(self, index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index][view_index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp.copy()

        border = self.border
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk, msk_cihp

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        params['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def prepare_input_image(self, pose_index, idx):
        img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk, origin_msk = np.array(self.get_mask(pose_index, idx))
        img[origin_msk == 0] = 0

        # Reduce the image resolution by ratio, then remove the back ground
        ratio = self.image_scaling
        if ratio != 1.:
            H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2,0,1))
        return img

    def __getitem__(self, index):

        instance_idx = index // (self.poses_num * self.camera_view_num) if self.multi_person else 0
        pose_index = ( index % (self.poses_num * self.camera_view_num) ) // self.camera_view_num #* self.poses_interval + self.poses_start
        view_index = index % self.camera_view_num

        self.data_root = self.all_humans[instance_idx]
        self.ims = self.ims_all[instance_idx]
        self.cams = self.cams_all[instance_idx]
        self.cam_inds = self.cam_inds_all[instance_idx]

        if pose_index >= len(self.ims):
            pose_index = np.random.randint(len(self.ims))

        img_all, ray_o_all, ray_d_all, near_all, far_all = [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, mask_at_box_large_all = [], [], []
        obs_img_all, obs_K_all, obs_R_all, obs_T_all = [], [], [], []

        # Load image, mask, K, D, R, T
        img_path = os.path.join(self.data_root, self.ims[pose_index][view_index].replace('\\', '/'))
        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk, origin_msk = np.array(self.get_mask(pose_index, view_index))

        cam_ind = self.cam_inds[pose_index][view_index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        img[origin_msk == 0] = 0

        # Reduce the image resolution by ratio, then remove the back ground
        ratio = self.image_scaling
        if ratio != 1.:
            H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * ratio

        i = int(os.path.basename(img_path)[:-4])
        feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)

        # Sample rays in target space world coordinate
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk = sample_ray_neubody_batch(
                img, msk, K, R, T, world_bounds, 1.0)

        mask_at_box_large = mask_at_box
        # Pack all inputs of all views
        img = np.transpose(img, (2,0,1))

        bkgd_msk[bkgd_msk != 0] = 1

        # target view
        img_all.append(img)
        ray_o_all.append(ray_o)
        ray_d_all.append(ray_d)
        near_all.append(near)
        far_all.append(far)
        mask_at_box_all.append(mask_at_box)
        bkgd_msk_all.append(bkgd_msk)
        mask_at_box_large_all.append(mask_at_box_large)

        # obs view data preparation
        if self.split == 'train':
            if self.sample_obs_view:
                self.obs_view_index = np.random.randint(self.camera_view_num)
            else:
                self.obs_view_index = 10

        if self.obs_pose_index is not None:
            obs_pose_index = int(self.obs_pose_index)
        else:
            obs_pose_index = pose_index

        # Load image, mask, K, D, R, T in observation space
        obs_img_path = os.path.join(self.data_root, self.ims[obs_pose_index][self.obs_view_index].replace('\\', '/'))
        obs_img = np.array(imageio.imread(obs_img_path).astype(np.float32) / 255.)
        obs_msk, origin_msk = np.array(self.get_mask(obs_pose_index, self.obs_view_index))

        obs_cam_ind = self.cam_inds[obs_pose_index][self.obs_view_index]
        obs_K = np.array(self.cams['K'][obs_cam_ind])
        obs_D = np.array(self.cams['D'][obs_cam_ind])

        # obs_img = cv2.undistort(obs_img, K, D)
        # obs_msk = cv2.undistort(obs_msk, K, D)

        obs_R = np.array(self.cams['R'][obs_cam_ind])
        obs_T = np.array(self.cams['T'][obs_cam_ind]) / 1000.
        obs_img[origin_msk == 0] = 0

        # Reduce the image resolution by ratio, then remove the back ground
        ratio = self.image_scaling
        if ratio != 1.:
            H, W = int(obs_img.shape[0] * ratio), int(obs_img.shape[1] * ratio)
            obs_img = cv2.resize(obs_img, (W, H), interpolation=cv2.INTER_AREA)
            obs_msk = cv2.resize(obs_msk, (W, H), interpolation=cv2.INTER_NEAREST)
            obs_K[:2] = obs_K[:2] * ratio

        obs_img = np.transpose(obs_img, (2,0,1))

        obs_i = int(os.path.basename(obs_img_path)[:-4])
        _, _, _, _, _, _, _, obs_vertices, obs_params = self.prepare_input(obs_i)

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

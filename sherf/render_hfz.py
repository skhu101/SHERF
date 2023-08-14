import os
import json
import imageio
import numpy as np
from smpl.smpl_numpy import SMPL
from cv2 import Rodrigues as rodrigues
from scipy.spatial.transform import Rotation as R


frame = 0
view = 0
# folder = 'data/RenderPeople/seq_000000_rp_aaron_rigged_001'
folder = 'data/RenderPeople/seq_000003_rp_adanna_rigged_001'

smpl_data = np.load(os.path.join(folder, 'smpl_new.npz')) # ['betas', 'global_orient', 'transl', 'body_pose']
img = imageio.imread(os.path.join(folder, 'img/{}/{}.jpg'.format(str(view).zfill(2), str(frame).zfill(4)))) # (1024, 1024, 3)
camera = json.load(open(os.path.join(folder, 'camera_param/camera{}.json'.format(str(view).zfill(2))), 'r')) # ['class_name', 'convention', 'extrinsic_r', 'extrinsic_t', 'height', 'intrinsic', 'name', 'width', 'world2cam']

smpl_beta = smpl_data['betas'].reshape(10)
smpl_theta = np.zeros([1, 72])
smpl_theta[0, 3:] = smpl_data['body_pose'][frame]
glob_ori_0 = R.from_rotvec(smpl_data['global_orient'][0])
glob_ori_curr = R.from_rotvec(smpl_data['global_orient'][frame])
glob_ori = glob_ori_curr / glob_ori_0
smpl_theta[0, :3] = glob_ori
smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
vert, _ = smpl_model(smpl_theta, smpl_beta)
# vert = np.matmul(vert, rodrigues(smpl_data['global_orient'][frame])[0].transpose()) + smpl_data['transl'][frame]
vert = vert + smpl_data['transl'][frame]

R = np.array(camera['extrinsic_r']).astype(np.float32)
T = np.array(camera['extrinsic_t']).astype(np.float32)
K = np.array([
    [512, 0, 512],
    [0, 512, 512],
    [0, 0, 1]
]).astype(np.float32)

vert = np.matmul(vert, R.transpose()) + T
uv = vert / vert[:, -1:]
uv = np.matmul(uv, K.transpose())

print(R)
print(T)
print(smpl_theta[:, :3])

for p in uv:
    y = int(p[0]); x = int(p[1])
    if x >= img.shape[0] or x < 0 or y >= img.shape[1] or y < 0:
        continue
    img[x-1:x+1, y-1:y+1] = 255

imageio.imwrite('tmp_img.png', img)

KITTI_CALIB_FILES_PATH="kitti/calibration/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH="kitti/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH="kitti/testing/image_3/*.png"

RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
BGNET_PLUS_MODEL_PATH = "pretrained_models/bgnet/kitti_15_BGNet_Plus.pth"
GWCNET_MODEL_PATH = "pretrained_models/gwcnet/gwcnet_g.ckpt"
PASMNET_MODEL_PATH = "pretrained_models/pasmnet/PASMnet_192_kitti_epoch5101.pth.tar"

CRESTEREO_MODEL_PATH = "pretrained_models/crestereo/crestereo_eth3d.pth"
HITNET_MODEL_PATH = "pretrained_models/hitnet/bestD1_checkpoint.ckpt"
PSMNET_MODEL_PATH = 'pretrained_models/psmnet/pretrained_model_KITTI2015.tar'

import torch
# Automatically detect if CUDA is available and set device accordingly
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DEVICE1 = "cuda:1"
else:
    DEVICE = "cpu"
    DEVICE1 = "cpu"


# raft-stereo=0, fastacv-plus=1, bgnet=2, gwcnet=3, pasmnet=4, crestereo=5, hitnet=6, psmnet=7

# gwcnet runs out of memory on my laptop (8GB)
ARCHITECTURE_LIST = ["raft-stereo", "fastacv-plus", "bgnet", 'gwcnet', 'pasmnet', 'crestereo', 'hitnet', 'psmnet']
ARCHITECTURE = ARCHITECTURE_LIST[7]
SAVE_POINT_CLOUD = 0
SHOW_DISPARITY_OUTPUT = 1
SHOW_3D_PROJECTION = 0

# Enable this Profile flag only when you want to Profile
PROFILE_FLAG = 0
PROFILE_IMAGE_WIDTH = 512
PROFILE_IMAGE_HEIGHT = 512

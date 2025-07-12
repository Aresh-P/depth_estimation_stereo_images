import os

import cv2
import time
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d

from utils import get_calibration_parameters, calc_depth_map, find_distances, add_depth, Open3dVisualizer, write_ply
from object_detector import ObjectDetectorAPI
from disparity_estimator.raftstereo_disparity_estimator import RAFTStereoEstimator
from disparity_estimator.fastacv_disparity_estimator import FastACVEstimator
from disparity_estimator.bgnet_disparity_estimator import BGNetEstimator
from disparity_estimator.gwcnet_disparity_estimator import GwcNetEstimator
from disparity_estimator.pasmnet_disparity_estimator import PASMNetEstimator
from disparity_estimator.crestereo_disparity_estimator import CREStereoEstimator
from disparity_estimator.psmnet_disparity_estimator import PSMNetEstimator
from disparity_estimator.hitnet_disparity_estimator import HitNetEstimator
import config



def get_disparity_estimator(architecture):
    """Create and return the appropriate disparity estimator based on architecture name."""
    if architecture == 'raft-stereo':
        return RAFTStereoEstimator()
    elif architecture == 'fastacv-plus':
        return FastACVEstimator()
    elif architecture == 'bgnet':
        return BGNetEstimator()
    elif architecture == 'gwcnet':
        return GwcNetEstimator()
    elif architecture == 'pasmnet':
        return PASMNetEstimator()
    elif architecture == 'crestereo':
        return CREStereoEstimator()
    elif architecture == 'psmnet':
        return PSMNetEstimator()
    elif architecture == 'hitnet':
        return HitNetEstimator()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def demo():
    if config.PROFILE_FLAG:
        disp_estimator = get_disparity_estimator(config.ARCHITECTURE)
        disp_estimator.profile()
        exit()

    left_images = sorted(glob.glob(config.KITTI_LEFT_IMAGES_PATH, recursive=True))
    right_images = sorted(glob.glob(config.KITTI_RIGHT_IMAGES_PATH, recursive=True))
    calib_files = sorted(glob.glob(config.KITTI_CALIB_FILES_PATH, recursive=True))
    index = 0
    init_open3d = False
    
    # Initialize with current architecture
    current_arch_idx = config.ARCHITECTURE_LIST.index(config.ARCHITECTURE)
    current_architecture = config.ARCHITECTURE
    disp_estimator = get_disparity_estimator(current_architecture)
    print("Disparity Architecture Used: {} ".format(current_architecture))
    print("\nKeyboard controls:")
    print("  'n' or RIGHT ARROW - Next model")
    print("  'p' or LEFT ARROW - Previous model") 
    print("  'q' or ESC - Quit")
    print("  '1-8' - Jump to specific model")
    print("")

    if config.SHOW_DISPARITY_OUTPUT:
        window_name = "Disparity Estimation Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for (imfile1, imfile2, calib_file) in tqdm(list(zip(left_images, right_images, calib_files))):
        img = cv2.imread(imfile1)
        parameters = get_calibration_parameters(calib_file)

        # Temporarily disable object detection due to TensorFlow compatibility issues
        # TODO: Fix YOLOv4 compatibility with TensorFlow 2.19
        result = img.copy()
        pred_bboxes = []
        print("Object detection temporarily disabled")

        start_d = time.time()
        disparity_map = disp_estimator.estimate(imfile1, imfile2)
        end_d = time.time()
        elapsed_time_d = (end_d - start_d) * 1000
        print("Evaluation Time for Disparity Estimation with {} is : {} ms ".format(config.ARCHITECTURE, elapsed_time_d))

        print("disparity_map: {}".format(disparity_map.shape))
        disparity_left = disparity_map

        k_left = parameters[0]
        t_left = parameters[1]
        p_left = parameters[2]

        k_right = parameters[3]
        t_right = parameters[4]
        p_right = parameters[5]
        print("k_left:{}, t_left:{}".format(k_left, t_left))
        print("k_right:{}, t_right:{}".format(k_right, t_right))
        print("p_left:{}, p_right:{}".format(p_left, p_right))

        depth_map = calc_depth_map(disparity_map, k_left, t_left, t_right)
        disparity_map = (disparity_map * 256.).astype(np.uint16)
        color_depth = cv2.applyColorMap(cv2.convertScaleAbs(disparity_map, alpha=0.01), cv2.COLORMAP_JET)

        depth_list = find_distances(depth_map, pred_bboxes, img, method="center")

        res = add_depth(depth_list, result, pred_bboxes)
        print("img.shape {}".format(img.shape))
        print("color_depth.shape {}".format(color_depth.shape))
        print("res.shape {}".format(res.shape))
        h = img.shape[0]
        w = img.shape[1]
        color_depth = cv2.resize(color_depth, (w, h))
        print("color_depth.shape after resize {}".format(color_depth.shape))
        combined_image = np.vstack((color_depth, res))
        if config.SHOW_DISPARITY_OUTPUT:
            cv2.imshow(window_name, combined_image)
        if config.SHOW_3D_PROJECTION:
            if init_open3d == False:
                w = img.shape[1]
                h = img.shape[0]
                print("w:{}, h: {}".format(w, h))
                print("kleft[0][0]: {}".format(k_left[0][0]))
                print("kleft[1][2]: {}".format(k_left[1][1]))
                print("kleft[1][2]: {}".format(k_left[0][2]))
                print("kleft[1][2]: {}".format(k_left[1][2]))
                print("kLeft: {}".format(k_left))

                K = o3d.camera.PinholeCameraIntrinsic(width=w,
                                                      height=h,
                                                      fx=k_left[0, 0],
                                                      fy=k_left[1, 1],
                                                      cx=k_left[0][2],
                                                      cy=k_left[1][2])
                open3dVisualizer = Open3dVisualizer(K)
                init_open3d = True
            open3dVisualizer(img, depth_map * 1000)

            o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
            o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
            o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat, cv2.COLOR_RGB2BGR)
        if config.SAVE_POINT_CLOUD:
            # Calculate depth-to-disparity
            cam1 = k_left  # left image - P2
            cam2 = k_right  # right image - P3

            print("p_left: {}".format(p_left))
            print("cam1:{}".format(cam1))

            Tmat = np.array([0.54, 0., 0.])
            Q = np.zeros((4, 4))
            cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                              distCoeffs1=0, distCoeffs2=0,
                              imageSize=img.shape[:2],
                              R=np.identity(3), T=Tmat,
                              R1=None, R2=None,
                              P1=None, P2=None, Q=Q)

            print("Disparity To Depth")
            print(Q)
            print("disparity_left.shape: {}".format(disparity_left.shape))
            print("disparity_left: {}".format(disparity_left))

            points = cv2.reprojectImageTo3D(disparity_left.copy(), Q)
            # reflect on x axis

            reflect_matrix = np.identity(3)
            reflect_matrix[0] *= -1
            points = np.matmul(points, reflect_matrix)

            img_left = cv2.imread(imfile1)
            colors = cv2.cvtColor(img_left.copy(), cv2.COLOR_BGR2RGB)
            print("colors.shape: {}".format(colors.shape))
            disparity_left = cv2.resize(disparity_left, (colors.shape[1], colors.shape[0]))
            points = cv2.resize(points, (colors.shape[1], colors.shape[0]))
            print("points.shape: {}".format(points.shape))
            print("After mod. disparity_left.shape: {}".format(disparity_left.shape))
            # filter by min disparity
            mask = disparity_left > disparity_left.min()
            out_points = points[mask]
            out_colors = colors[mask]

            out_colors = out_colors.reshape(-1, 3)
            path_ply = os.path.join("output/point_clouds/", config.ARCHITECTURE)
            isExist = os.path.exists(path_ply)
            if not isExist:
                os.makedirs(path_ply)
            print("path_ply: {}".format(path_ply))

            file_name = path_ply + "/" +str(index) + ".ply"
            print("file_name: {}".format(file_name))
            write_ply(file_name, out_points, out_colors)
            index = index + 1
        if config.SHOW_DISPARITY_OUTPUT:
            # Add model info to the image
            info_text = f"Model: {current_architecture} (Press 'n' for next, 'p' for previous, 'q' to quit)"
            cv2.putText(combined_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, combined_image)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == 83:  # 'n' or right arrow
                # Switch to next model
                current_arch_idx = (current_arch_idx + 1) % len(config.ARCHITECTURE_LIST)
                current_architecture = config.ARCHITECTURE_LIST[current_arch_idx]
                print(f"\nSwitching to: {current_architecture}")
                disp_estimator = get_disparity_estimator(current_architecture)
                # Re-process current image with new model
                print("Reprocessing current image with new model...")
                continue
            elif key == ord('p') or key == 81:  # 'p' or left arrow  
                # Switch to previous model
                current_arch_idx = (current_arch_idx - 1) % len(config.ARCHITECTURE_LIST)
                current_architecture = config.ARCHITECTURE_LIST[current_arch_idx]
                print(f"\nSwitching to: {current_architecture}")
                disp_estimator = get_disparity_estimator(current_architecture)
                # Re-process current image with new model
                print("Reprocessing current image with new model...")
                continue
            elif ord('1') <= key <= ord('8'):  # Number keys 1-8
                # Direct model selection
                model_idx = key - ord('1')
                if 0 <= model_idx < len(config.ARCHITECTURE_LIST):
                    current_arch_idx = model_idx
                    current_architecture = config.ARCHITECTURE_LIST[current_arch_idx]
                    print(f"\nSwitching to: {current_architecture}")
                    disp_estimator = get_disparity_estimator(current_architecture)
                    print("Reprocessing current image with new model...")
                    continue
    if config.SHOW_DISPARITY_OUTPUT:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    demo()



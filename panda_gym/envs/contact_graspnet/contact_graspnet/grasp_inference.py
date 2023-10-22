import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from panda_gym.envs.contact_graspnet.contact_graspnet import config_utils
from panda_gym.envs.contact_graspnet.contact_graspnet import data

from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from scipy.spatial.transform import Rotation as R

class CGNInference:
    def __init__(self):
        checkpoint_dir = '/host/panda_gym/envs/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001'
        global_config = config_utils.load_config(checkpoint_dir, batch_size=1, arg_configs=[])
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')
        
        os.makedirs('results', exist_ok=True)

    def run_inference(self, points, colors):
        pc_full = points
        pc_colors = colors

        rot = R.from_euler('xy',[90,90], degrees=True)
        pc_full = (rot.as_matrix()@pc_full.T).T

        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments=None, 
                                                                                          local_regions=None, filter_grasps=False, forward_passes=1)  


        #grasp_point, grasp_rot, approach_point = visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        #grasp_point = rot.inv().as_matrix()@grasp_point
        #approach_point = rot.inv().as_matrix()@approach_point

        grasp_points, grasp_rots, approach_points,best_idx = visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        grasp_points = rot.inv().as_matrix()@grasp_points.T
        approach_points = rot.inv().as_matrix()@approach_points.T
        return grasp_points.T, grasp_rots, approach_points.T, best_idx

if __name__ == '__main__':
    inf = CGNInference()

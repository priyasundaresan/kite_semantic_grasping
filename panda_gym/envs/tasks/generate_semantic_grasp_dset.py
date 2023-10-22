from typing import Any, Dict
from collections.abc import Iterable
import random
import time
import os

import numpy as np
import cv2
import open3d as o3d

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial
from sklearn.metrics.pairwise import euclidean_distances

class TableTop:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.body_id_mapping = {}

    def _create_scene(self) -> None:
        """Create the scene."""
        #num_objs = np.random.randint(3,6)
        num_objs = 1

        grasping_locs = self.reset_sim(num_objs)

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        # Delete graspable objects
        for k in self.body_id_mapping:
            if 'body' in k:
                self.sim.physics_client.removeBody(self.body_id_mapping[k])
                try:
                    self.sim._bodies_idx.pop(k)
                except:
                    continue

        # Put graspable objects in scene
        self.graspable_obj_names = []
        self.graspable_objs = []
        for i in range(len(grasping_locs)):
            fn = random.choice(os.listdir('grasping_assets'))
            #fn = random.choice(os.listdir('grasp_assets'))
            self.graspable_obj_names.append(fn)
                
            ori = R.from_euler('xyz', [0,0,np.random.uniform(0,180)], degrees=True).as_quat()
            obj = self.sim.loadURDF(body_name='body%d'%i, fileName='grasping_assets/%s/model.urdf'%fn, basePosition=grasping_locs[i], baseOrientation=ori, globalScaling=0.85)
            #obj = self.sim.loadURDF(body_name='body%d'%i, fileName='grasp_assets/%s/model.urdf'%fn, basePosition=grasping_locs[i], baseOrientation=ori, globalScaling=0.85)

            self.graspable_objs.append('body%d'%i)
            self.body_id_mapping['body%d'%i] = obj
        self.graspable_obj_names = self.filter_names(self.graspable_obj_names)

    def filter_names(self, fns):
        filtered_names = []
        for fn in fns:
            words = fn.split('_')
            if len(words) == 1:
                filtered_names.append(words[0])
            else:
                if words[-1].isdigit():
                    words = words[:-1]
                if words[0] in ['plastic']:
                    words = words[1:]
                filtered_names.append(' '.join(words))
        return filtered_names

    def dist(self, new_point, points, r_threshold):
        for point in points:
            dist = np.sqrt(np.sum(np.square(new_point-point)))
            if dist < r_threshold:
                return False
        return True
    
    def RandX(self,N, r_threshold):
        points = []
        while len(points) < N:
            #new_point = np.random.uniform([-0.3,-0.3,0.075], [0.0,0.1,0.075])
            new_point = np.random.uniform([-0.1,-0.1,0.075], [0.0,0.1,0.075])
            if self.dist(new_point, points, r_threshold):
                points.append(new_point)
        return points

    def reset_sim(self, num_locs=3):
        filtered_points = self.RandX(num_locs, 0.075)
        return filtered_points

    def reset_robot(self):
        self.robot.reset()
        goal_euler_xyz = np.array([180,0,0]) # standard
        self.robot.move(np.array([0,0,0.6]), goal_euler_xyz)
        self.robot.release()

    def reset(self):
        with self.sim.no_rendering():
            self._create_scene()
        #self.reset_robot()
        for i in range(10):
            self.sim.step()

    def take_rgbd(self):
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        _, _, cam2world = self.sim.get_cam2world_transforms(target_position=np.array([-0.1,0.1,0]), distance=0.9, yaw=90, pitch=-70)
        img, depth, points, colors, pixels_2d, waypoints_proj = self.sim.render(target_position=np.zeros(3), distance=0.4, yaw=90, pitch=-45)
        return img

    def record(self, img, episode_idx):
        cv2.imwrite('dset/images/%05d.jpg'%(episode_idx), img)
        cv2.imshow('img', img)
        cv2.waitKey(30)
        prompt = 'Pick up the %s by the '%(self.graspable_obj_names[0])
        prompt += input(prompt + '____?')
        np.save('dset/lang/%05d.npy'%(episode_idx), prompt)

if __name__ == '__main__':
    dset_dir = 'dset'
    images_dir = os.path.join(dset_dir, 'images')
    lang_dir = os.path.join(dset_dir, 'lang')
    for d in [dset_dir, images_dir, lang_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = TableTop(sim, robot)

    episode = 0

    while episode < 20:
        if episode%20 == 0:
            task.sim.close()
            sim = PyBullet(render=True, background_color=np.array([255,255,255]))
            robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
            task = TableTop(sim, robot)

        task.reset()
        img = task.take_rgbd()
        task.record(img, episode)
        episode += 1

from typing import Any, Dict
import random
import time
import os
import open3d as o3d

import numpy as np
import cv2

#from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from panda_gym.envs.contact_graspnet.contact_graspnet.grasp_inference import CGNInference
from panda_gym.envs.inference.inference_kpt import KptInference

from sklearn.neighbors import NearestNeighbors

class Manipulate:
    def __init__(
        self,
        sim: PyBullet,
        robot,
    ) -> None:
        self.sim = sim
        self.robot = robot
        #self._create_scene()
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.inference_server = CGNInference()
        self.kpt_inference_server = KptInference(checkpoint_start='checkpoint_grasp/model.pth')
        self.body_id_mapping = {}
        self.prompt_dir = 'semantic_grasping_dset'

    def load_instruction(self):
        fn = random.choice(os.listdir(os.path.join(self.prompt_dir, 'lang')))
        instr = np.load(os.path.join(self.prompt_dir, 'lang', fn))
        instr = str(instr)
        graspable_object = '_'.join(instr.split('Pick up the')[1].split('by the')[0].strip().split())
        object_part = instr.split('Pick up the')[1].split('by the')[1].strip()
        return graspable_object, object_part

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
        self.graspable_obj_parts = []
        self.graspable_objs = []


        for i in range(len(grasping_locs)):
            graspable_object, object_part = self.load_instruction()

            #fn = random.choice(os.listdir('semantic_assets'))

            #self.graspable_obj_names.append(fn)
            self.graspable_obj_names.append(graspable_object)
            self.graspable_obj_parts.append(object_part)
                
            ori = R.from_euler('xyz', [0,0,np.random.uniform(0,180)], degrees=True).as_quat()
            obj = self.sim.loadURDF(body_name='body%d'%i, fileName='semantic_assets/%s/model.urdf'%graspable_object, basePosition=grasping_locs[i], baseOrientation=ori, globalScaling=0.875)

            self.graspable_objs.append('body%d'%i)
            self.body_id_mapping['body%d'%i] = obj

        #self.graspable_obj_names = self.filter_names(self.graspable_obj_names)

    def reset(self):
        with self.sim.no_rendering():
            success = False
            while not success:
                try:
                    self._create_scene()
                    success = True
                except:
                    continue
        for i in range(10):
            self.sim.step()


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

    def take_rgbd(self):
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        _, _, cam2world = self.sim.get_cam2world_transforms(target_position=np.zeros(3), distance=0.4, yaw=90, pitch=-45)
        img, depth, points, colors, pixels_2d, waypoints_proj = self.sim.render(target_position=np.zeros(3), distance=0.4, yaw=90, pitch=-45)

        #img, depth, points, colors, pixels_2d, _ = self.robot.sim.render(distance=1.2, yaw=45, pitch=-30)

        _, _, points1, colors1, pixels1_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=0)
        _, _, points2, colors2, pixels_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=135)
        _, _, points3, colors3, pixels_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=245)
        _, _, points4, colors4, pixels_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=300)

        points = np.vstack((points1, points2, points3, points4))
        colors = np.vstack((colors1, colors2, colors3, colors4))

        return img, points, colors, depth, cam2world


    def visualize(self, points, colors):
        pcd = o3d.geometry.PointCloud()
    
        rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
        rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
        points = (rot@points.T).T
    
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/255.)
        o3d.visualization.draw_geometries([pcd])

    def deproject(self, pixel, depth, cam2world):
        point = self.sim.deproject(depth, np.array(pixel).reshape(-1,2), cam2world)
        return point

    def execute(self, episode):
        save_path = 'preds/%05d.jpg'%episode

        self.reset_robot()
        img, points, colors, depth, cam2world = self.take_rgbd()

        #prompt = 'Pick up the %s by the '%(self.graspable_obj_names[0])
        #inp = input(prompt + '____?')
        #prompt += inp

        prompt = 'Pick up the %s by the %s'%(self.graspable_obj_names[0], self.graspable_obj_parts[0])

        semantic_waypoint_proj = self.kpt_inference_server.run_inference(img, prompt, save_path=save_path)
    
        semantic_waypoint  = self.deproject(semantic_waypoint_proj, depth, cam2world)[0]

        x_low = semantic_waypoint[0] - 0.125
        x_high = semantic_waypoint[0] + 0.125
        y_low = semantic_waypoint[1] - 0.125
        y_high = semantic_waypoint[1] + 0.125

        idxs = np.where(points[:,0] > x_low)[0]
        points = points[idxs]
        colors = colors[idxs]
        idxs = np.where(points[:,0] < x_high)[0]
        points = points[idxs]
        colors = colors[idxs]
        idxs = np.where(points[:,1] > y_low)[0]
        points = points[idxs]
        colors = colors[idxs]
        idxs = np.where(points[:,1] < y_high)[0]
        points = points[idxs]
        colors = colors[idxs]

        #self.visualize(points, colors)

        grasp_positions, angles, approaches, best_idx = self.inference_server.run_inference(points, colors)

        nbrs = NearestNeighbors(n_neighbors=1).fit(grasp_positions)
        dists, idxs  = nbrs.kneighbors(semantic_waypoint.reshape(1,-1), return_distance=True)
        best_idx = idxs[0][0]
        
        grasp_pos = grasp_positions[best_idx]
        (yaw,pitch,roll) = angles[best_idx]
        approach_pos = approaches[best_idx]

        offset = approach_pos - grasp_pos
        approach_pos = grasp_pos + 2*offset
        grasp_pos -= 0.3*offset

        lift_pos = grasp_pos + np.array([0,0,0.15])

        reset_euler = np.array([180.,0.,0.]) # standard

        grasp_euler = reset_euler + [yaw,pitch,roll]

        self.robot.move(approach_pos, grasp_euler)
        for i in range(10):
            self.sim.step()

        self.robot.move(grasp_pos, grasp_euler)

        for i in range(10):
            self.sim.step()

        self.robot.grasp()

        for i in range(10):
            self.sim.step()

        self.robot.move(lift_pos, grasp_euler)

        for i in range(10):
            self.sim.step()

        return ctr

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim, robot)

    if not os.path.exists('preds'):
        os.mkdir('preds')

    episode = 0
    for i in range(15):
        task.reset()
        try:
            episode = task.execute(episode)
        except:
            continue

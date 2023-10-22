from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray, euler_xyz=None) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement, euler_xyz)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            #fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_ctrl = action[-1]
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def get_waypoint(self, start_pt, target_pt, max_delta, num_steps=None):
        total_delta = (target_pt - start_pt)
        if num_steps is None:
            num_steps = (np.linalg.norm(total_delta) // max_delta)
            remainder = (np.linalg.norm(total_delta) % max_delta)
            if remainder > 1e-3:
                num_steps += 1

        delta = total_delta / num_steps
        def gen_waypoint(i):
            return start_pt + delta * min(i, num_steps)
        return gen_waypoint, int(num_steps)

    def get_ori(self, initial_euler, final_euler, num_steps):
        ori_chg = R.from_euler('xyz', [initial_euler.copy(), final_euler.copy()], degrees=True)
        slerp = Slerp([1,num_steps], ori_chg)
        def gen_ori(i): 
            #interp_quat = slerp(i).as_quat()
            #return interp_quat
            interp_euler = slerp(i).as_euler('xyz', degrees=True)
            return interp_euler
        return gen_ori

    #def move(self, start_pos, start_euler, goal_pos, goal_euler):
    def move(self, goal_pos, goal_euler):
        start_pos = self.get_ee_position()
        start_euler = self.get_ee_orientation()
        finger_width = self.get_fingers_width()
        #if finger_width < 0.1:
        #    finger_width = 0

        distance = np.linalg.norm(goal_pos - start_pos)
        if distance < 0.03:
            gen_ori_fn = self.get_ori(start_euler, goal_euler, 20)
            gen_fn, num_steps = self.get_waypoint(start_pos, goal_pos, 0.015, num_steps=20)
            #gen_fn, num_steps = self.get_waypoint(start_pos, goal_pos, 0.01, num_steps=20)

        else:
            #gen_fn, num_steps = self.get_waypoint(start_pos, goal_pos, 0.01)
            gen_fn, num_steps = self.get_waypoint(start_pos, goal_pos, 0.015)
            gen_ori_fn = self.get_ori(start_euler, goal_euler, num_steps)

        for i in range(1, num_steps+1):
            next_pos = gen_fn(i)
            next_euler = gen_ori_fn(i)
            action = (next_pos - self.get_ee_position())
            action = action.tolist() + [finger_width]
            self.set_action(action, next_euler)
            self.sim.step()

    def grasp(self):
        self.block_gripper = True
        #for i in range(100):
        for i in range(30):
            action = [0,0,0,1]
            self.set_action(action, self.get_ee_orientation())
            self.sim.step()

    #def grasp(self):
    #    self.block_gripper = False
    #    for i in range(5):
    #        action = [0,0,0,0.2]
    #        self.set_action(action, self.get_ee_orientation())
    #        self.sim.step()

    def release(self, width=1):
        self.block_gripper = False
        #for i in range(100):
        for i in range(30):
            action = [0,0,0,width]
            self.set_action(action, self.get_ee_orientation())
            self.sim.step()

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray, euler_xyz=None) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        #ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        ee_displacement = ee_displacement[:3] * 1  # limit maximum change in position

        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles

        standard_ori = np.array([1.0, 0.0, 0.0, 0.0])
        if euler_xyz is not None:
            ori = R.from_euler('xyz', euler_xyz, degrees=True).as_quat()
        else:
            ori = standard_ori

        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=ori
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        link_quat = self.get_link_orientation(self.ee_link)
        link_euler = R.from_quat(link_quat).as_euler('xyz', degrees=True)
        return link_euler

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

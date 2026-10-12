#!/usr/bin/env python
import os
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R

from configs import EnvContrConfiguration

yaml_path = os.path.join(
    pathlib.Path(__file__).resolve().parents[2], "config", "env_controller.yaml"
)
env_config = EnvContrConfiguration.from_yaml(yaml_path)
obs_num = env_config.num_dyn_circle
model_name = "actor"
# 1.3 or 0
default_z = 0.05


# circler
if env_config.env_name == "hospital":
    obs_position_list = np.zeros((5, 20, 3))

    k = 0
    obs_position_list[k, 0] = [-6, -5.50, -np.pi]
    obs_position_list[k, 1] = [-10, -5.50, -3 / 2 * np.pi]
    obs_position_list[k, 2] = [-10, -2.7, -2 * np.pi]
    obs_position_list[k, 3] = [-6, -2.7, -3 / 2 * np.pi]
    obs_position_list[k, 4] = [-6, 2.7, -3 / 2 * np.pi]
    obs_position_list[k, 5] = [-6, 5, -np.pi]
    obs_position_list[k, 6] = [-10, 5, -1 / 2 * np.pi]
    obs_position_list[k, 7] = [-10, 2, -np.pi / 2]
    obs_position_list[k, 8] = [-10, -2, -np.pi / 2]
    obs_position_list[k, 9] = [-10, -5.5, -np.pi / 2]
    obs_position_list[k, 10] = [-6, -5.50, -np.pi]
    obs_position_list[k, 11] = [-10, -5.50, -3 / 2 * np.pi]
    obs_position_list[k, 12] = [-10, -5.50, -3 / 2 * np.pi]

    k = 1
    obs_position_list[k, 0] = [-15, 8, -np.pi / 2]
    obs_position_list[k, 1] = [-15, 5.5, 0]
    obs_position_list[k, 2] = [-10.4, 5.5, 0]
    obs_position_list[k, 3] = [-10.4, 5.5, -np.pi / 2]
    obs_position_list[k, 4] = [-10.2, 1.5, -np.pi / 2]
    obs_position_list[k, 5] = [-10, -2.5, -np.pi / 2]
    obs_position_list[k, 6] = [-10, -5.5, -np.pi / 2]
    obs_position_list[k, 7] = [-10, -5.5, 0]
    obs_position_list[k, 8] = [-6, -5.5, 0]
    obs_position_list[k, 9] = [-6, -5.50, -np.pi]
    obs_position_list[k, 10] = [-10, -5.50, -np.pi]
    obs_position_list[k, 11] = [-10, -5.50, -3 / 2 * np.pi]
    obs_position_list[k, 12] = [-10, -2, -3 / 2 * np.pi]
    obs_position_list[k, 13] = [-10, -2, -2 * np.pi]
    obs_position_list[k, 14] = [-6, -2, -2 * np.pi]
    obs_position_list[k, 15] = [-6, -2, -3 / 2 * np.pi]
    obs_position_list[k, 16] = [-6, 2, -3 / 2 * np.pi]

    # nurses
    k = 2
    obs_position_list[k, 0] = [-16, -10, np.pi / 2]
    obs_position_list[k, 1] = [-16, -6, np.pi / 2]
    obs_position_list[k, 2] = [-16, -2, np.pi / 2]
    obs_position_list[k, 3] = [-16, 2, np.pi / 2]
    obs_position_list[k, 4] = [-16, 5.5, np.pi / 2]
    obs_position_list[k, 5] = [-15, 5.5, np.pi / 2]
    obs_position_list[k, 6] = [-15, 9, np.pi / 2]
    obs_position_list[k, 7] = [-15, 9, np.pi / 2]
    obs_position_list[k, 8] = [-15, 9, -np.pi / 2]
    obs_position_list[k, 9] = [-15, 5.5, -np.pi / 2]
    obs_position_list[k, 10] = [-16, 5.5, -np.pi / 2]
    obs_position_list[k, 11] = [-16, 2, -np.pi / 2]
    obs_position_list[k, 12] = [-16, -2, -np.pi / 2]
    obs_position_list[k, 13] = [-16, -6, -np.pi / 2]
    obs_position_list[k, 14] = [-16, -10, -np.pi / 2]
    obs_position_list[k, 15] = [-16, -10, 0]
    obs_position_list[k, 16] = [-16, -10, 0]

    obs_position_list[3, 0] = [-2.5, 0.3, -np.pi / 2]
    obs_position_list[3, 1] = [-6.5, 0.4, -np.pi / 2]
    obs_position_list[3, 2] = [-11, 0.4, -np.pi / 2]
    obs_position_list[3, 3] = [-11, 0.4, -np.pi]
    obs_position_list[3, 4] = [-11, 2.6, -np.pi]
    obs_position_list[3, 5] = [-11, 4.8, -3 * np.pi / 2]
    obs_position_list[3, 6] = [-8.1, 4.8, -3 * np.pi / 2]
    obs_position_list[3, 7] = [-8.1, 5.5, -np.pi]
    obs_position_list[3, 8] = [-8.1, 5.5, -np.pi]
    obs_position_list[3, 9] = [-8.1, 5.5, -3 * np.pi / 2]
    obs_position_list[3, 10] = [-5, 5.5, -3 / 2 * np.pi]
    obs_position_list[3, 11] = [-5, 5.5, -2 * np.pi]
    obs_position_list[3, 12] = [-5, 1.5, -2 * np.pi]
    obs_position_list[3, 13] = [-5, 0.4, -5 / 2 * np.pi]
    obs_position_list[3, 14] = [-9.5, 0.4, -5 / 2 * np.pi]
    obs_position_list[3, 15] = [-9.5, 0.4, -5 / 2 * np.pi]
    obs_position_list[3, 16] = [-10.5, 0.4, -5 / 2 * np.pi]
    obs_position_list[3, 17] = [-10.5, 0.4, -5 / 2 * np.pi]

    # male vistor
    k = 4
    obs_position_list[k, 0] = [-4.5, -5.50, -3 / 2 * np.pi]
    obs_position_list[k, 1] = [-4.5, -1.50, -3 / 2 * np.pi]
    obs_position_list[k, 2] = [-4.5, 2.50, -3 / 2 * np.pi]
    obs_position_list[k, 3] = [-6, 6, -3 / 2 * np.pi]
    obs_position_list[k, 4] = [-6, 9, -3 / 2 * np.pi]
    obs_position_list[k, 5] = [-6, 9, -3 / 2 * np.pi]
    obs_position_list[k, 6] = [-6, 9, -3 / 2 * np.pi]
elif env_config.env_name == "social_nav_dynamic":
    obs_position_list = np.zeros((10, 20, 3))
    mid_pos = np.zeros((env_config.num_dyn_circle, 3))
    mid_pos[0] = [1, -2.5, 1.57]
    mid_pos[1] = [0.2, -1.8, 1.57]
    mid_pos[2] = [3, -0.3, 3.14]
    mid_pos[3] = [3, 0.3, 3.14]
    mid_pos[4] = [0.3, 3, -2]
    mid_pos[5] = [1, 3, -1.5]
    mid_pos[6] = [2, 2, -1.5]
    mid_pos[7] = [2.5, 0.7, 3.14]
    mid_pos[8] = [2.5, -0.7, 3.14]
    mid_pos[9] = [0.7, 3, -2]

    v_pos = np.zeros((env_config.num_dyn_circle, 3))
    for k in range((env_config.num_dyn_circle)):
        v_pos[k, :2] = [np.cos(mid_pos[k, 2]), np.sin(mid_pos[k, 2])]
    for k in range(env_config.num_dyn_circle):
        for i in range(20):
            obs_position_list[k, i] = mid_pos[k] + 0.5 * v_pos[k] * i


def x_c(iter, steps=200, dt=0.05):
    # steps: how many iterations the human take to go from position at time t to that at time t+1
    t = min(int(iter / steps), obs_position_list.shape[1] - 2)
    vxy = dx_c(iter, steps, dt)
    x_center = obs_position_list[:, t, :] + dt * vxy * (iter % steps)
    return x_center


def dx_c(iter, steps=200, dt=0.05):
    # steps: how many iterations the human take to go from position at time t to that at time t+1
    t = int(iter / steps)
    if t > obs_position_list.shape[1] - 2:
        return np.zeros(obs_position_list[:, 0, :].shape)
    linear_v = (obs_position_list[:, t + 1, :] - obs_position_list[:, t, :]) / (
        dt * steps
    )
    return linear_v

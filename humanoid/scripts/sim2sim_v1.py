import sys

sys.path.append(f'~/miniconda3/envs/humanoid_gym/lib/python3.8/site-packages')

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
import torch

import copy


class cmd:
    not_stand = 1.0
    vx = 0.3
    vy = 0.0
    dyaw = 0.0


from scipy.optimize import fsolve

KA = 0.236;
PA = 0.0582;
ang_PAB = 1.4893;
l1 = 0.1056;
l4 = 0.0839;
l3 = 0.0812;
l2 = 0.0753;
l0 = 0.2273466101
theta3_0 = 0.005684626950805134


def theta3_to_l(theta3_target, initial_guess=None):
    KA = 0.236
    PA = 0.0582
    ang_PAB = 1.4893
    l1 = 0.1056
    l4 = 0.0839
    l3 = 0.0812
    l2 = 0.0753

    # 默认初始猜测基于正解数据范围
    if initial_guess is None:
        initial_guess = [0.24, 0.5]  # 典型初始值

    def equations(x):
        l, theta2 = x
        # 计算ang_KAP和theta1
        numerator = KA ** 2 + PA ** 2 - l ** 2
        denominator = 2 * KA * PA

        # 处理数值误差导致的无效输入
        if denominator == 0:
            return [np.inf, np.inf]
        cos_ang = numerator / denominator
        cos_ang = np.clip(cos_ang, -1.0, 1.0)  # 避免数值误差
        ang_KAP = np.arccos(cos_ang)
        theta1 = np.pi - ang_KAP - ang_PAB

        # 四杆机构闭环方程
        eq1 = l1 * np.cos(theta1) + l2 * np.cos(theta2) - l3 * np.cos(theta3_target) - l4
        eq2 = l1 * np.sin(theta1) - l2 * np.sin(theta2) - l3 * np.sin(theta3_target)
        return [eq1, eq2]

    # 数值求解
    solution, info, ier, msg = fsolve(equations, initial_guess, full_output=True)
    # 收敛检查
    # if ier != 1:
    #     raise ValueError(f"未收敛: theta3={theta3_target:.4f}, 错误信息: {msg}")
    l_solution = solution[0]
    # 二次验证解的合理性
    if not (0.2 <= l_solution <= 0.3):
        print(f"警告: theta3={theta3_target:.4f}时得到异常值l={l_solution:.6f}")
    return l_solution


def dtheta3_dl(l):
    # 常数定义
    KA = 0.236
    PA = 0.0582
    ang_PAB = 1.4893
    l1 = 0.1056
    l4 = 0.0839
    l3 = 0.0812
    l2 = 0.0753

    # 计算 ang_KAP 和 theta1
    ang_KAP = np.arccos((KA ** 2 + PA ** 2 - l ** 2) / (2 * KA * PA))
    theta1 = np.pi - ang_KAP - ang_PAB

    # 计算 A, B, C
    A = 2 * l1 * l3 * np.cos(theta1) - 2 * l3 * l4
    B = 2 * l1 * l3 * np.sin(theta1)
    C = l1 ** 2 - l2 ** 2 + l3 ** 2 + l4 ** 2 - 2 * l1 * l4 * np.cos(theta1)
    theta3 = np.arctan2(B, A) - np.arccos(C / np.sqrt(A ** 2 + B ** 2))

    # 求 ang_KAP 对 l 的导数
    d_ang_KAP_dl = 2 * l / np.sqrt((2 * KA * PA) ** 2 - (KA ** 2 + PA ** 2 - l ** 2) ** 2)

    # 求 theta1 对 l 的导数
    dtheta1_dl = -d_ang_KAP_dl

    # 对 A, B, C 分别求导
    dA_dl = 2 * l1 * l3 * (-np.sin(theta1)) * dtheta1_dl
    dB_dl = 2 * l1 * l3 * np.cos(theta1) * dtheta1_dl
    dC_dl = 2 * l1 * l4 * np.sin(theta1) * dtheta1_dl

    # 对 theta3 求导数
    dtheta3_dl = (A * dB_dl - B * dA_dl) / (A ** 2 + B ** 2)  # 对 arctan2(B, A) 求导
    # 对 -arccos(C / np.sqrt(A^2 + B^2)) 求导
    dtheta3_dl += (dC_dl / np.sqrt(A ** 2 + B ** 2) - (C * (A * dA_dl + B * dB_dl)) / (A ** 2 + B ** 2) ** (
                3 / 2)) / np.sqrt(1 - (C / np.sqrt(A ** 2 + B ** 2)) ** 2)

    return dtheta3_dl


def length_to_theta(l1, l2):
    # 定义方程系统的函数
    def equations(theta):
        theta_y, theta_x = theta

        # A1 - ankle_roll
        o_p_a1 = np.array([0.007, -0.03, 0.261])
        o_p_a2 = np.array([0.007, 0.031, 0.261])

        # B1 - ankle_roll
        a_p_b1 = np.array([-0.078, -0.03, 0.])
        a_p_b2 = np.array([-0.078, 0.031, 0.])

        # 计算o_p_b1和o_p_b2的位置
        o_p_b1 = np.array([np.cos(theta_y) * a_p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * a_p_b1[1],
                           np.cos(theta_x) * a_p_b1[1],
                           -np.sin(theta_y) * a_p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * a_p_b1[1]])

        o_p_b2 = np.array([np.cos(theta_y) * a_p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * a_p_b2[1],
                           np.cos(theta_x) * a_p_b2[1],
                           -np.sin(theta_y) * a_p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * a_p_b2[1]])

        # 计算l1和l2
        l1_calc = np.linalg.norm(o_p_a1 - o_p_b1)
        l2_calc = np.linalg.norm(o_p_a2 - o_p_b2)

        # 返回方程的误差值，期望误差为零
        return [l1_calc - l1, l2_calc - l2]

    # 初始猜测值（theta_y, theta_x）
    theta_initial = np.array([0.0, 0.0])

    # 使用fsolve求解方程
    theta_solution, info, ier, msg = fsolve(equations, theta_initial, full_output=True)

    # if ier != 1:
    #     raise ValueError(f"未收敛: l1={l1:.4f}, l2={l2:.4f} 错误信息: {msg}")

    # 返回求解得到的theta_y和theta_x
    return theta_solution[0], theta_solution[1]


def theta_to_length(theta_y, theta_x):
    # A1 - ankle_roll
    o_p_a1 = np.array([0.007, -0.03, 0.261])
    o_p_a2 = np.array([0.007, 0.031, 0.261])

    # B1 - ankle_roll
    a_p_b1 = np.array([-0.078, -0.03, 0.])
    a_p_b2 = np.array([-0.078, 0.031, 0.])

    o_p_b1 = np.array([np.cos(theta_y) * a_p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * a_p_b1[1],
                       np.cos(theta_x) * a_p_b1[1],
                       -np.sin(theta_y) * a_p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * a_p_b1[1]])

    o_p_b2 = np.array([np.cos(theta_y) * a_p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * a_p_b2[1],
                       np.cos(theta_x) * a_p_b2[1],
                       -np.sin(theta_y) * a_p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * a_p_b2[1]])

    l1 = np.linalg.norm(o_p_a1 - o_p_b1)
    l2 = np.linalg.norm(o_p_a2 - o_p_b2)

    return l1, l2


def Jacobian_dtheta_dl_differential(theta_y, theta_x):
    delta_l = 1e-5
    l1, l2 = theta_to_length(theta_y, theta_x)
    theta_y_11, _ = length_to_theta(l1 + delta_l, l2)
    theta_y_12, _ = length_to_theta(l1, l2 + delta_l)
    _, theta_x_21 = length_to_theta(l1 + delta_l, l2)
    _, theta_x_22 = length_to_theta(l1, l2 + delta_l)
    J11 = (theta_y_11 - theta_y) / delta_l
    J12 = (theta_y_12 - theta_y) / delta_l
    J21 = (theta_x_21 - theta_x) / delta_l
    J22 = (theta_x_22 - theta_x) / delta_l
    return np.array([[J11, J12], [J21, J22]])



def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def Jacobian_dl_dtheta(theta_y, theta_x):
    # A1 - ankle_roll
    o_p_a1 = np.array([0.007, -0.03, 0.261])
    o_p_a2 = np.array([0.007, 0.031, 0.261])

    # B1 - ankle_roll
    p_b1 = np.array([-0.078, -0.03, 0.])
    p_b2 = np.array([-0.078, 0.031, 0.])

    o_p_b1 = np.array([np.cos(theta_y) * p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * p_b1[1],
                       np.cos(theta_x) * p_b1[1],
                       -np.sin(theta_y) * p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * p_b1[1]])

    o_p_b2 = np.array([np.cos(theta_y) * p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * p_b2[1],
                       np.cos(theta_x) * p_b2[1],
                       -np.sin(theta_y) * p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * p_b2[1]])

    l1 = np.linalg.norm(o_p_a1 - o_p_b1)
    l2 = np.linalg.norm(o_p_a2 - o_p_b2)

    dl1_dtheta_y = ((np.cos(theta_y) * p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * p_b1[1] - o_p_a1[0]) * (
                -np.sin(theta_y) * p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * p_b1[1]) -
                    (-np.sin(theta_y) * p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * p_b1[1] - o_p_a1[2]) * (
                                np.cos(theta_y) * p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * p_b1[1])) / l1

    dl1_dtheta_x = ((np.cos(theta_y) * p_b1[0] + np.sin(theta_y) * np.sin(theta_x) * p_b1[1] - o_p_a1[0]) * (
                np.sin(theta_y) * np.cos(theta_x) * p_b1[1]) +
                    (np.cos(theta_x) * p_b1[1] - o_p_a1[1]) * (-np.sin(theta_x) * p_b1[1]) +
                    (-np.sin(theta_y) * p_b1[0] + np.cos(theta_y) * np.sin(theta_x) * p_b1[1] - o_p_a1[2]) * (
                                np.cos(theta_y) * np.cos(theta_x) * p_b1[1])) / l1

    dl2_dtheta_y = ((np.cos(theta_y) * p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * p_b2[1] - o_p_a2[0]) * (
                -np.sin(theta_y) * p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * p_b2[1]) -
                    (-np.sin(theta_y) * p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * p_b2[1] - o_p_a2[2]) * (
                                np.cos(theta_y) * p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * p_b2[1])) / l2

    dl2_dtheta_x = ((np.cos(theta_y) * p_b2[0] + np.sin(theta_y) * np.sin(theta_x) * p_b2[1] - o_p_a2[0]) * (
                np.sin(theta_y) * np.cos(theta_x) * p_b2[1]) +
                    (np.cos(theta_x) * p_b2[1] - o_p_a2[1]) * (-np.sin(theta_x) * p_b2[1]) +
                    (-np.sin(theta_y) * p_b2[0] + np.cos(theta_y) * np.sin(theta_x) * p_b2[1] - o_p_a2[2]) * (
                                np.cos(theta_y) * np.cos(theta_x) * p_b2[1])) / l2

    return np.array([[dl1_dtheta_y, dl1_dtheta_x],
                     [dl2_dtheta_y, dl2_dtheta_x]])


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('Body_Gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # mujoco.mj_resetDataKeyframe(model, data, 0)
    # mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body {i}: {body_name}")

    target_q = np.zeros((12), dtype=np.double)
    action = np.zeros((12), dtype=np.double)

    hist_obs = deque()
    for _ in range(15):
        hist_obs.append(np.zeros([1, 47], dtype=np.double))

    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q_dof = np.concatenate((q[7:7 + 6], q[7 + 6 + 24:7 + 6 + 24 + 6]), axis=0) - cfg.robot_config.default_dof_pos
        dq_dof = np.concatenate((dq[6:6 + 6], dq[6 + 6 + 20:6 + 6 + 20 + 6]), axis=0)

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, 47], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 2] = cmd.vx * 1.0
            obs[0, 3] = cmd.vy * 1.0
            obs[0, 4] = cmd.dyaw * 1.0

            obs[0, 5:17] = q_dof * 1.0
            obs[0, 17:29] = dq_dof * 0.05
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -18, 18)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, 15 * 47], dtype=np.float32)
            for i in range(15):
                policy_input[0, i * 47: (i + 1) * 47] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -18., 18.)

            target_q = action * 1.0

        target_dq = np.zeros((12), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q_dof, cfg.robot_config.kps,
                         target_dq, dq_dof, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        tau[3] *= -dtheta3_dl(theta3_to_l(q_dof[3] + cfg.robot_config.default_dof_pos[3]))
        tau[9] *= -dtheta3_dl(theta3_to_l(q_dof[9] + cfg.robot_config.default_dof_pos[9]))

        ankle_yt_l = tau[4].copy()
        ankle_xt_l = tau[5].copy()
        ankle_yt_r = tau[10].copy()
        ankle_xt_r = tau[11].copy()

        # 方法一： 通过数值求解的方式计算雅可比矩阵
        # J_L = -Jacobian_dtheta_dl_differential(-(q_dof[4]+cfg.robot_config.default_dof_pos[4]),
        #                                     q_dof[5]+cfg.robot_config.default_dof_pos[5]).transpose()
        # J_R = -Jacobian_dtheta_dl_differential(-(q_dof[10]+cfg.robot_config.default_dof_pos[10]),
        #                                     q_dof[11]+cfg.robot_config.default_dof_pos[11]).transpose()
        # # tau[4:6] = np.dot(J_L, tau[4:6])
        # # tau[10:12] = np.dot(J_R, tau[10:12])
        # tau[4] = -J_L[0,0] * ankle_yt_l + J_L[0,1] * ankle_xt_l
        # tau[5] = -J_L[1,0] * ankle_yt_l + J_L[1,1] * ankle_xt_l
        # tau[10] = -J_R[0,0] * ankle_yt_r + J_R[0,1] * ankle_xt_r
        # tau[11] = -J_R[1,0] * ankle_yt_r + J_R[1,1] * ankle_xt_r

        # 方法二： 解析解求解雅可比矩阵
        J_L_dl_dtheta = -Jacobian_dl_dtheta(-(q_dof[4] + cfg.robot_config.default_dof_pos[4]),
                                            q_dof[5] + cfg.robot_config.default_dof_pos[5])
        J_R_dl_dtheta = -Jacobian_dl_dtheta(-(q_dof[10] + cfg.robot_config.default_dof_pos[10]),
                                            q_dof[11] + cfg.robot_config.default_dof_pos[11])
        tau[4] = -ankle_yt_l / J_L_dl_dtheta[0, 0] + ankle_xt_l / J_L_dl_dtheta[0, 1]
        tau[5] = -ankle_yt_l / J_L_dl_dtheta[1, 0] + ankle_xt_l / J_L_dl_dtheta[1, 1]
        tau[10] = -ankle_yt_r / J_R_dl_dtheta[0, 0] + ankle_xt_r / J_R_dl_dtheta[0, 1]
        tau[11] = -ankle_yt_r / J_R_dl_dtheta[1, 0] + ankle_xt_r / J_R_dl_dtheta[1, 1]
        print('tuili', dq_dof[0], dq_dof[1], dq_dof[2])
        # 写入txt文本数据
        # move_data = ''
        # for j in range(len(q_dof)):
        #     data_ = round(q_dof[j], 3)
        #     move_data += str(data_)
        #     move_data += ','
        # if count_lowlevel > 4000:
        #     move_data += str(1.0)
        # else:
        #     move_data += str(0.0)
        # with open('move_data.txt', 'a') as file:
        #     line = move_data + "\n"
        #     file.write(line)
        #     file.flush()
        # file.close()

        #读取txt文本数据
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/robot18/gymloong/sim2sim/Humanoid-Gym-main'
                                                          '/humanoid/scripts/policies/12joints.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = ('../../resources/robots/long/mujoco_model/Robot_12_Cloosedloop_V02.xml')
            else:
                mujoco_model_path = ('../../resources/robots/long/mujoco_model/Robot_12_Cloosedloop_V02.xml')
            sim_duration = 500.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # kps = np.array([100,100,120,160,12,12,100,100,120,160,12,12], dtype=np.double)
            kps = np.array([100, 100, 120, 160, 12, 12, 100, 100, 120, 160, 12, 12], dtype=np.double)
            kds = np.array([10.0,10.0,12.0,16.0,1.2,1.2,10.0,10.0,12.0,16.0,1.2,1.2], dtype=np.double)
            default_dof_pos = np.array([0,0,0.0,0,0,0, 0,0,0.0,0.0,0.0,0])
            tau_limit = np.array([200,200,300,500,200,200, 200,200,300,500,200,200], dtype=np.double)


    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())

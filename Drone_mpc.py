import airsim
import numpy as np
import argparse
from utils import Statistics
from sac_ae_empc import SAC_Ae_eMpc
from sac_ae_qmpc import SAC_Ae_qMpc
import logging
from Logger import Logger
from utils import generate_points
from torch.utils.tensorboard import SummaryWriter
import os
import time
import socket
from mpi4py import MPI
from datetime import datetime
from Environment import Environment
from Mpcstructure import MpcStructure
import utils
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'dsac.json')
with open(_CONFIG_PATH, 'r') as _f:
    _config = json.load(_f)

args = argparse.Namespace(
    policy=_config.get("policy", "SAC_Ae_apf"),
    num_agent=_config.get("num_agent", 1),
    num_barrier=_config.get("num_barrier", 0),
    seed=_config.get("seed", 0),
    num_layers=_config.get("num_layers", 4),
    num_filters=_config.get("num_filters", 32),
    log_std_min=_config.get("log_std_min", -10.0),
    log_std_max=_config.get("log_std_max", 2.0),
    batch_size=_config.get("batch_size", 512),
    replayer_buffer=_config.get("replayer_buffer", 80000),
    discount=_config.get("discount", 0.99),
    tau=_config.get("tau", 0.01),
    learning_rate=_config.get("learning_rate", 0.0001),
    max_episodes=_config.get("max_episodes", 2001),
    max_timesteps=_config.get("max_timesteps", 200),
    episode_step=_config.get("episode_step", 20),
    init_steps=_config.get("init_steps", 500),
    obs_shape=_config.get("obs_shape", [30]),
    action_shape=_config.get("action_shape", 1),
    action_shape_2=_config.get("action_shape_2", 1),
    hidden_dim=_config.get("hidden_dim", 512),
    lam_a=_config.get("lam_a", -1),
    lam_s=_config.get("lam_s", -1),
    eps_s=_config.get("eps_s", 0.2),
    mode=_config.get("mode", "train"),
    encoder_type=_config.get("encoder_type", "identity"),
    decoder_type=_config.get("decoder_type", "identity"),
    encoder_feature_dim=_config.get("encoder_feature_dim", 30),
    dt=_config.get("dt", 0.2),
    show_predicted_trajectory=_config.get("show_predicted_trajectory", True),
    fixed_points=_config.get("fixed_points", True),
    log_reward=_config.get("log_reward", True),
)


# 为 Q 权重调整 policy 创建 kwargs
kwargs_q = {
    "seed": args.seed,                              # 随机种子
    "batch_size": args.batch_size,                  # 批量大小
    "replayer_buffer": args.replayer_buffer,       # 经验回放容量
    "obs_shape": args.obs_shape,                    # 状态维度
    "num_env": args.num_agent,                      # 环境中无人机数量
    "action_shape": args.action_shape,             # 动作维度
    "discount": args.discount,                      # SAC折扣因子
    "tau": args.tau,                                # 目标网络软更新比例
    "lr": args.learning_rate,                       # 学习率
    "hidden_dim": args.hidden_dim,                  # 隐藏层神经元数
    "init_steps": args.init_steps,                  # 初始化随机探索步数
    "mode": args.mode,                              # 'train' 或 'test'
    # Actor 相关
    "num_layers": args.num_layers,                  # Actor隐藏层数量
    "num_filters": args.num_filters,                # 编码器特征数（占位）
    # "log_std_min": args.log_std_min,                # Actor log_std 最小值
    # "log_std_max": args.log_std_max,                # Actor log_std 最大值
}

# 为 MPC 更新控制 policy 创建 kwargs
kwargs_mpc = {
    "seed": args.seed,
    "batch_size": args.batch_size,
    "replayer_buffer": args.replayer_buffer,
    "obs_shape": args.obs_shape,
    "num_env": args.num_agent,
    "action_shape": args.action_shape_2,
    "discount": args.discount,
    "tau": args.tau,
    "lr": args.learning_rate,
    "hidden_dim": args.hidden_dim,
    "init_steps": args.init_steps,
    "mode": args.mode,
    # Actor 相关
    "num_layers": args.num_layers,
    "num_filters": args.num_filters,
    # "log_std_min": args.log_std_min,
    # "log_std_max": args.log_std_max,
}

def convert_env_state_to_mpc_state(env):
    """
    将环境状态转换为MPC控制器所需的状态格式
    
    参数:
    env: Environment实例
    
    返回:
    mpc_state: [x, y, z, yaw, vx, wz, vz] (7维状态向量)
    """
    # 获取位置
    position = env.get_state(return_index=1)  # [x, y, z]
    
    # 获取速度信息
    state = env.client.getMultirotorState(vehicle_name="Drone_" + str(env.index))
    angular_vel = state.kinematics_estimated.angular_velocity
    local_goal, local_speed = env.get_local_goal_and_speed()
    
    # 将四元数转换为欧拉角，只取yaw角
    quat = state.kinematics_estimated.orientation
    roll, pitch, yaw = airsim.to_eularian_angles(quat)
    
    # 显式获取全局 Z 轴线速度
    global_vz = state.kinematics_estimated.linear_velocity.z_val
    
    # 构建MPC状态向量
    mpc_state = np.array([
        position[0],  # x
        position[1],  # y
        position[2],  # z
        yaw,          # yaw
        local_speed[0],  # vx (机体坐标系前进速度)
        angular_vel.z_val,  # wz (偏航角速度)
        global_vz     # 全局 global_vz
    ])
    
    return mpc_state,local_goal

def run(comm, env, policy_q=None, policy_mpc=None, starting_epoch=0, train_policy_q=False, train_policy_mpc=False, statistics=None):
    sucess_list = []
    rewardq_list = []
    rewardmpc_list = []

    # 参数
    Q = [0.5, 0.5, 3.0,   # 位置权重
         0.5, 0.5, 0.5,   # 控制平滑权重
         0.5]             # 障碍物权重

    horizon = 15
    max_velocity = 0.5
    max_acceleration = 1.0
    max_yaw_acceleration = 1.0
    max_yaw_rate = 1.0

    # 速度归一化初始化
    min_vel_bounds = [-max_velocity, -max_yaw_rate, -max_velocity] 
    max_vel_bounds = [ max_velocity, max_yaw_rate, max_velocity]
    velocity_rms = utils.RunningMeanStd(min_vel_bounds, max_vel_bounds)

    # MPC控制参数
    mpc_update_frequency = 1  # 每N步更新一次MPC，1表示每步都更新，2表示每2步更新一次
    trigger_steps_before_end = 9  # 控制序列剩余步数低于此值时强制触发
    
    # 初始化MPC控制器
    mpc_controller = MpcStructure(
        Q=Q,
        dt=env.dt,  # 采样时间
        horizon=horizon,  # 预测时域长度
        max_velocity=max_velocity,  # 最大速度限制
        max_acceleration=max_acceleration,  # 最大加速度限制
        max_yaw_acceleration=max_yaw_acceleration,  # 最大角加速度限制
        max_yaw_rate=max_yaw_rate  # 最大角速度限制
    )
    
    mpc_controller.update_weights(weight=Q)
    env.get_obstacles()

    # MPC控制相关变量
    last_mpc_control_sequence = None
    last_mpc_index = 0
    mpc_step_counter = 0

    # 在循环外判断模型是否加载，避免每步重复判断
    use_policy_mpc = policy_mpc is not None
    use_policy_q = policy_q is not None

    # 无模型时的默认值（固定不变）
    action_mpc = [1.0]
    action_q = [0.5]

    env.start_polling(interval=0.01)

    for epoch in range(starting_epoch, args.max_episodes):
        
        env.client.simPause(False)
        terminal = False
        next_episode = False
        liveflag = True
        ep_rewardq = 0
        ep_rewardmpc = 0
        step = 1
        exp_q = None
        exp_mpc = None
        result = ''
        
        # 重置MPC控制变量
        last_mpc_control_sequence = None
        last_mpc_index = 0
        mpc_step_counter = 0
        dynamic_Q = [0.5, 0.5, 3.0, 0.5, 0.5, 0.5, 0.5]  # 默认Q权重
        
        # 统计相关变量
        episode_mpc_updates = 0  # 本episode的MPC更新次数
        episode_mpc_failures = 0  # 本episode的MPC失败次数
        episode_mpc_reuses = 0  # 本episode的MPC重用次数
        episode_mpc_computations = 0  # 本episode的MPC计算次数
        episode_start_pos = None  # 起始位置
        episode_total_distance = 0.0  # 总飞行距离
        episode_accels = []  # 记录本episode中MPC给出的加速度序列，用于计算RMS Jerk
        all_episodes_mpc_time = []
        episode_total_mpc_time = 0.0
        
        # 逐步序列记录
        episode_trajectory = []    # 每步位置 [(x,y,z), ...]
        episode_velocity = []
        episode_acceleration = []
        episode_trigger_seq = []   # 每步MPC触发信号 [0/1, ...]
        episode_dist_to_goal = []  # 每步到目标距离 [d, ...]
        episode_q_weights = []     # 每步Q权重 [[q0..q6], ...]
        
        # 获取初始位置并初始化“上一帧”变量
        init_pos_val = env.get_state(return_index=1) # [x, y, z]
        last_pos_array = np.array(init_pos_val, dtype=np.float32)
        last_vel = np.zeros(3, dtype=np.float32) # 假设初始速度为0
        
        # 根据训练的policy设置各自的epoch
        if train_policy_q and policy_q is not None:
            policy_q.epoch = epoch
        if train_policy_mpc and policy_mpc is not None:
            policy_mpc.epoch = epoch
        if run_test_flag:
            # 测试模式下，两个policy都使用当前epoch
            if policy_q is not None:
                policy_q.epoch = epoch
            if policy_mpc is not None:
                policy_mpc.epoch = epoch
        
        # 生成起始点与终点
        pose_list, goal_list = generate_points(num_env=args.num_agent, map_index=env.map_index, fixed=args.fixed_points)

        # 1. 最先重置世界环境 (清理上一回合的状态)
        env.reset_world()
        env.current_epoch = epoch 
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]
        
        # 临时构建一个 Pose 对象用于底层强制瞬移，无视碰撞
        temp_pose = airsim.Pose()
        temp_pose.position.x_val = pose_ctrl[0]
        temp_pose.position.y_val = pose_ctrl[1]
        temp_pose.position.z_val = pose_ctrl[2]
        env.client.simSetVehiclePose(temp_pose, ignore_collision=True, vehicle_name="Drone_" + str(env.index))
        
        # 给仿真器一点时间完成物理引擎的重置和空间跳跃
        time.sleep(2)
        env.drones_init()
        time.sleep(1)
        
        # 获取当前实际稳定后的位姿，然后生成终点并最终校准姿态
        init_pose = env.get_state()
        env.generate_goal_point(goal_ctrl)
        env.reset_pose(init_pose, pose_ctrl)
        current_mpc_state, _ = convert_env_state_to_mpc_state(env)
        
        # 记录起始位置用于计算飞行距离
        episode_start_pos = env.get_state(return_index=1)
        last_pos_array = np.array(episode_start_pos, dtype=np.float32)
        last_vel = np.zeros(3, dtype=np.float32)
        
        # 构造强化学习输入
        O_z = np.array(env.get_obstacle_vector(), dtype=np.float32)             # (30,)
        O_yaw = np.array([env.get_state(return_index=6)], dtype=np.float32)     # (1,)
        O_q = np.array([float(0.5)], dtype=np.float32)                          # (1,)
        O_velocity = np.array(current_mpc_state[4:], dtype=np.float32)          # (3,)

        # 构建状态向量
        state_q = [O_z, O_q, O_velocity]
        state_mpc = [O_z, O_yaw, O_velocity]

        while not next_episode:   
            # 使用MPC生成速度控制指令
            
            if liveflag:

                step_start_time = time.time()
                agent2_wanted_update = True
                env.move_obstacles(step)

                try:
                    # 事件触发规则
                    # 1、剩余序列不足
                    rule_triggered = False
                    if last_mpc_control_sequence is None:
                        rule_triggered = True
                    elif (len(last_mpc_control_sequence) - last_mpc_index) <= trigger_steps_before_end:
                        rule_triggered = True
                    
                    # 2、速度过低时强制触发，防止无人机停滞
                    current_speed = np.sqrt(current_mpc_state[4]**2 + current_mpc_state[6]**2)
                    if current_speed < 0.3:
                        rule_triggered = True
                    
                    # 3、智能体判断触发
                    if use_policy_mpc:
                        try:
                            actions_mpc = policy_mpc.generate_n_action(env, [state_mpc], velocity_rms)
                            action_mpc = actions_mpc[0]
                            if action_mpc is not None and len(action_mpc) > 0:
                                mpc_update_threshold = 0.5
                                should_update_mpc = float(action_mpc[0]) > mpc_update_threshold
                                agent2_wanted_update = should_update_mpc 
                            else:
                                should_update_mpc = True
                                agent2_wanted_update = True 
                                action_mpc = [1.0]
                        except Exception:
                            should_update_mpc = True
                            agent2_wanted_update = True 
                            action_mpc = [1.0]
                    else:
                        # 无模型：固定频率触发
                        should_update_mpc = (mpc_step_counter % mpc_update_frequency == 0) or (last_mpc_control_sequence is None)
                        agent2_wanted_update = should_update_mpc
                    
                    # 规则强制覆盖：rule_triggered 时无论 Agent2 输出什么都必须触发
                    # agent2_wanted_update 已在覆盖前保存，不受影响
                    if rule_triggered:
                        should_update_mpc = True
                        
                    if should_update_mpc:
                        # 重新计算MPC控制序列                        
                        current_mpc_state, _ = convert_env_state_to_mpc_state(env)
                        global_goal = np.array(env.goal)  # 全局目标位置 [x, y, z]

                        # Q权重：有模型则推理，无模型直接用默认值
                        if use_policy_q:
                            try:
                                actions_q = policy_q.generate_q_action(env, [state_q], velocity_rms)
                                action_q = np.array(actions_q).flatten()
                                if action_q is not None and len(action_q) > 0:
                                    dynamic_Q = [
                                        float(0.5),  # 位置权重 x
                                        float(0.5),  # 位置权重 y
                                        float(3.0),  # 位置权重 z
                                        float(0.5),  # 控制平滑 ax
                                        float(0.5),  # 控制平滑 aw
                                        float(0.5),  # 控制平滑 az
                                        float(action_q[0])  # 障碍物代价权重
                                    ]
                                    mpc_controller.update_weights(weight=dynamic_Q)
                                    env.Q = dynamic_Q
                                    if disp_info_flag:
                                        print(f"Q权重Policy更新Q权重: {dynamic_Q}")
                                else:
                                    action_q = [0.5]
                            except Exception as e:
                                if disp_info_flag:
                                    print(f"Q权重Policy获取Q权重失败: {e}，使用默认Q权重")
                                action_q = [0.5]
                        
                        # 使用MPC生成控制指令
                        episode_mpc_computations += 1
                        mpc_solve_start = time.time()
                        mpc_control_sequence, success = mpc_controller.get_velocity_command(
                            current_mpc_state, 
                            global_goal, 
                            obstacles=env.obstacles
                        )
                        mpc_solve_time = time.time() - mpc_solve_start 
                        episode_total_mpc_time += mpc_solve_time
                        
                        if success and mpc_control_sequence is not None:
                            episode_mpc_updates += 1
                            last_mpc_control_sequence = mpc_control_sequence
                            last_mpc_index = 0
                            mpc_velocity = mpc_control_sequence[0]
                        else:
                            episode_mpc_failures += 1
                            if last_mpc_control_sequence is not None:
                                if disp_info_flag:
                                    print(f'MPC optimization failed, using last successful control sequence in step {step}')
                                episode_mpc_reuses += 1
                                if last_mpc_index < len(last_mpc_control_sequence) - 1:
                                    last_mpc_index += 1
                                mpc_velocity = last_mpc_control_sequence[last_mpc_index]
                            else:
                                # 没有历史控制序列时使用pid指令
                                if disp_info_flag:
                                    print('MPC optimization failed and no previous sequence, using fallback')
                                last_mpc_control_sequence = None
                                current_pos = current_mpc_state[:3]
                                target_pos = global_goal
                                pos_error = target_pos - current_pos
                                distance = np.linalg.norm(pos_error)
                                target_yaw = np.arctan2(pos_error[1], pos_error[0])
                                current_yaw = current_mpc_state[3]
                                yaw_error = np.arctan2(np.sin(target_yaw - current_yaw), 
                                                    np.cos(target_yaw - current_yaw))
                                current_vx = current_mpc_state[4]
                                current_wz = current_mpc_state[5]
                                current_vz = current_mpc_state[6]
                                max_speed = 1.0
                                desired_speed = min(max_speed, max(0.1, distance / 3.0))
                                if abs(yaw_error) > np.pi/4:
                                    desired_speed *= 0.3
                                height_error = pos_error[2]
                                desired_vz = np.clip(height_error * 0.5, -0.5, 0.5)
                                yaw_rate_gain = 0.8
                                desired_wz = np.clip(yaw_error * yaw_rate_gain, -1.0, 1.0)
                                current_speed_in_target_dir = current_vx * np.cos(target_yaw) + current_vz * np.sin(target_yaw)
                                speed_error = desired_speed - current_speed_in_target_dir
                                accel_gain = 0.5
                                ax_desired = np.clip(speed_error * accel_gain, -0.5, 0.5)
                                az_desired = desired_vz
                                aw_desired = desired_wz
                                mpc_velocity = np.array([ax_desired, aw_desired, az_desired])
                                if disp_info_flag:
                                    print(f'Step {step}: MPC优化失败，使用fallback控制 - 距离:{distance:.2f}m, yaw误差:{np.degrees(yaw_error):.1f}°, 控制:{mpc_velocity}')
                    else:
                        # 使用上一次控制序列的下一个指令
                        if last_mpc_control_sequence is not None and last_mpc_index < len(last_mpc_control_sequence) - 1:
                            episode_mpc_reuses += 1
                            last_mpc_index += 1
                            mpc_velocity = last_mpc_control_sequence[last_mpc_index]
                            if disp_info_flag:
                                print(f"Step {step}: 使用上一次控制序列，索引: {last_mpc_index}")
                        else:
                            # 控制序列用完了，强制重新计算
                            rule_triggered = True
                            agent2_wanted_update = False
                            if disp_info_flag:
                                print(f"Step {step}: 控制序列用完，重新计算MPC")
                            
                            current_mpc_state,_ = convert_env_state_to_mpc_state(env)
                            global_goal = np.array(env.goal)

                            # Q权重：有模型则推理，无模型直接用默认值
                            if use_policy_q:
                                try:
                                    actions_q = policy_q.generate_q_action(env, [state_q], velocity_rms)
                                    action_q = np.array(actions_q).flatten()
                                    if action_q is not None and len(action_q) > 0:
                                        dynamic_Q = [
                                            float(0.5),
                                            float(0.5),
                                            float(2.0),
                                            float(0.5),
                                            float(0.5),
                                            float(0.5),
                                            float(action_q[0])
                                        ]
                                        mpc_controller.update_weights(weight=dynamic_Q)
                                        env.Q = dynamic_Q
                                    else:
                                        action_q = [0.5]
                                except Exception as e:
                                    if disp_info_flag:
                                        print(f"Q权重Policy获取Q权重失败: {e}，使用默认Q权重")
                                    action_q = [0.5]
                            
                            episode_mpc_computations += 1
                            mpc_solve_start = time.time()
                            mpc_control_sequence, success = mpc_controller.get_velocity_command(
                                current_mpc_state, 
                                global_goal, 
                                obstacles=env.obstacles
                            )
                            mpc_solve_time = time.time() - mpc_solve_start 
                            episode_total_mpc_time += mpc_solve_time
                            
                            if success and mpc_control_sequence is not None:
                                episode_mpc_updates += 1
                                last_mpc_control_sequence = mpc_control_sequence
                                last_mpc_index = 0
                                mpc_velocity = mpc_control_sequence[0]
                            else:
                                episode_mpc_failures += 1
                                if last_mpc_control_sequence is not None:
                                    if disp_info_flag:
                                        print('MPC optimization failed when sequence exhausted, using last successful control sequence')
                                    episode_mpc_reuses += 1
                                    mpc_velocity = last_mpc_control_sequence[-1]
                                else:
                                    velocity = [0.0, 0.0, 0.0]
                                    if disp_info_flag:
                                        print('MPC optimization failed and no previous sequence, using zero velocity command')
                                    last_mpc_control_sequence = None
                    
                    # 计算最终的控制指令
                    if last_mpc_control_sequence is not None:
                        current_mpc_state,_ = convert_env_state_to_mpc_state(env)
                        current_vx = current_mpc_state[4]
                        current_wz = current_mpc_state[5]
                        current_vz = current_mpc_state[6]
                        dt = mpc_controller.dt
                        new_vx = current_vx + mpc_velocity[0] * dt
                        new_wz = current_wz + mpc_velocity[1] * dt
                        new_vz = current_vz + mpc_velocity[2] * dt
                        episode_accels.append(np.array(mpc_velocity, dtype=float))
                        velocity_magnitude = np.sqrt(new_vx**2 + new_vz**2)
                        if velocity_magnitude > mpc_controller.max_velocity:
                            scale = mpc_controller.max_velocity / velocity_magnitude
                            new_vx *= scale
                            new_vz *= scale
                        if abs(new_wz) > mpc_controller.max_yaw_rate:
                            new_wz = np.sign(new_wz) * mpc_controller.max_yaw_rate
                        velocity = [new_vx, new_wz, new_vz]
                    
                    mpc_step_counter += 1
                        
                except Exception as e:
                    velocity = [0.0, 0.0, 0.0]
                    if disp_info_flag:
                        print(f'MPC control error: {e}') 
                
                # 预测并绘制MPC轨迹
                if args.show_predicted_trajectory and last_mpc_control_sequence is not None:
                    remaining_steps = len(last_mpc_control_sequence) - last_mpc_index
                    if remaining_steps > 0:
                        plot_sequence = np.zeros_like(last_mpc_control_sequence)
                        plot_sequence[:remaining_steps] = last_mpc_control_sequence[last_mpc_index:]
                        predicted_trajectory = mpc_controller.predict_trajectory(
                            current_mpc_state, plot_sequence
                        )
                        env.plot_predicted_trajectory(
                            predicted_trajectory[:remaining_steps + 1],
                            color_rgba=[0.0, 1.0, 0.0, 1.0],
                            thickness=8,
                            duration=0.2
                        )

                # 应用控制指令
                # 与环境和交互
                env.control_vel(velocity)
                if args.show_predicted_trajectory:
                    env.client.simFlushPersistentMarkers()

                # 控制指令发出后立即检测碰撞（读轮询线程缓存，无额外 API 开销）
                is_crashed = env.get_crash_state()

                # 取 Agent1 输出的第7维避障权重
                q_weight = float(action_q[0]) if action_q is not None else 1.0
                rewardq, rewardmpc, terminal, result = env.get_reward_terminate_result(
                    step,
                    update_flag=agent2_wanted_update,
                    rule_triggered=rule_triggered,
                    q_weight=q_weight,
                    is_crashed=is_crashed
                )
                env.record_pos.append(env.get_state())

                # 逐步统计
                current_pos_array = np.array(env.get_state(return_index=1), dtype=np.float32)
                step_distance = float(np.linalg.norm(current_pos_array - last_pos_array))
                episode_total_distance += step_distance

                current_vel = (current_pos_array - last_pos_array) / env.dt
                current_acc = (current_vel - last_vel) / env.dt
                last_pos_array = current_pos_array.copy()
                last_vel = current_vel.copy()

                episode_trajectory.append(current_pos_array.tolist())
                episode_velocity.append(current_vel.tolist())
                episode_acceleration.append(current_acc.tolist())
                episode_trigger_seq.append(1 if agent2_wanted_update else 0)
                episode_dist_to_goal.append(float(np.linalg.norm(current_pos_array - np.array(env.goal[:3]))))
                episode_q_weights.append([float(v) for v in dynamic_Q])

                if terminal:
                    # 碰撞或超时：用当前状态填充 next_state，避免采集无效的碰撞后状态
                    next_O_z = O_z.copy() if O_z is not None else np.zeros(30, dtype=np.float32)
                    next_O_yaw = O_yaw.copy() if O_yaw is not None else np.zeros(1, dtype=np.float32)
                    next_O_q = O_q.copy() if O_q is not None else np.zeros(1, dtype=np.float32)
                    next_O_velocity = O_velocity.copy() if O_velocity is not None else np.zeros(3, dtype=np.float32)
                    next_mpc_state = current_mpc_state
                else:
                    try:
                        next_O_z = np.array(env.get_obstacle_vector(), dtype=np.float32)
                    except Exception as e:
                        print(f'obstacle vector error: {e}')
                        next_O_z = np.zeros(30, dtype=np.float32)

                    # 获取下一个MPC状态
                    next_mpc_state, _ = convert_env_state_to_mpc_state(env)
                    next_O_yaw = np.array([env.get_state(return_index=6)], dtype=np.float32)
                    next_O_q = np.array([dynamic_Q[6]], dtype=np.float32)
                    next_O_velocity = np.array(next_mpc_state[4:], dtype=np.float32)

                next_state_q = [next_O_z, next_O_q, next_O_velocity]
                next_state_mpc = [next_O_z, next_O_yaw, next_O_velocity]

                not_done = 1. - float(terminal)
                ep_rewardq += rewardq   
                ep_rewardmpc += rewardmpc
                
                exp_q = None
                exp_mpc = None
                if train_policy_q and policy_q is not None:
                    # qMPC 经验：严格对应 9 个字段 (无 goal)
                    exp_q = [
                        np.array(O_z, dtype=np.float32),
                        np.array(O_q, dtype=np.float32).reshape(1,),
                        np.array(O_velocity, dtype=np.float32),
                        np.array(action_q, dtype=np.float32).reshape(1,),
                        np.array(next_O_z, dtype=np.float32),
                        np.array(next_O_q, dtype=np.float32).reshape(1,),
                        np.array(next_O_velocity, dtype=np.float32), 
                        float(rewardq),
                        float(not_done)
                    ]
                if train_policy_mpc and policy_mpc is not None:
                    # eMPC 经验：严格对应 9 个字段 (无 goal)
                    exp_mpc = [
                        np.array(O_z, dtype=np.float32),
                        np.array(O_yaw, dtype=np.float32),
                        np.array(O_velocity, dtype=np.float32),
                        np.array(action_mpc, dtype=np.float32),
                        np.array(next_O_z, dtype=np.float32),
                        np.array(next_O_yaw, dtype=np.float32),
                        np.array(next_O_velocity, dtype=np.float32),
                        float(rewardmpc),
                        float(not_done)
                    ]
            else:
                next_state_q = None
                next_state_mpc = None
                next_O_z = None
                next_O_q = None
                next_O_velocity = None
            
            if terminal:
                liveflag = False
                next_episode = True
                if result == 'Reach Goal':
                    if (len(sucess_list)) == 200:
                        sucess_list.pop(0)
                    sucess_list.append(1)
                else:
                    if (len(sucess_list)) == 200:
                        sucess_list.pop(0)
                    sucess_list.append(0)
            
            # next state 滚动
            state_q = next_state_q
            state_mpc = next_state_mpc
            O_z = next_O_z
            O_yaw = next_O_yaw    
            O_q = next_O_q        
            O_velocity = next_O_velocity
            current_mpc_state = next_mpc_state

            elapsed_time = time.time() - step_start_time
            if elapsed_time < args.dt:
                time.sleep(args.dt - elapsed_time)

            step += 1
            env.step = step
            if train_policy_q and policy_q is not None and exp_q is not None:
                policy_q.step([exp_q.copy()])
            if train_policy_mpc and policy_mpc is not None and exp_mpc is not None:
                policy_mpc.step([exp_mpc.copy()])

        all_episodes_mpc_time.append(episode_total_mpc_time)
        logger.info('Env %02d, Goal (%05.1f, %05.1f, %05.1f), Episode %05d, step %03d, Reward1 %-5.1f, '
                    'Reward2 %-5.1f, %s, map%d, sucessrate:%02.2f' %
                    (env.index, env.goal[0], env.goal[1], env.goal[2], epoch + 1, step, ep_rewardq, ep_rewardmpc,
                     result, env.map_index, np.sum(sucess_list) / len(sucess_list)))

        if len(rewardq_list) == 100:
            rewardq_list.pop(0)
        rewardq_list.append(ep_rewardq)
        if len(rewardmpc_list) == 100:
            rewardmpc_list.pop(0)

        rewardmpc_list.append(ep_rewardmpc)
        env.plot_path()

        if disp_info_flag:
            print('Rewardq %-5.1f, Rewardmpc %-5.1f, Result %s' % (ep_rewardq, ep_rewardmpc, result))
        env.client.simPause(False)
        
        # 在测试模式下记录统计数据
        if statistics is not None:
            # 确定是否成功
            success = (result == 'Reach Goal')
            # 确定结果类型
            if 'collision' in result.lower() or 'crash' in result.lower():
                result_type = 'collision'
            elif 'timeout' in result.lower() or 'time' in result.lower():
                result_type = 'timeout'
            elif success:
                result_type = 'success'
            else:
                result_type = 'other'

            # 计算直线距离 L_straight（用于额外距离 ED）
            if episode_start_pos is not None and hasattr(env, 'goal') and len(env.goal) >= 3:
                start_pos_arr = np.array(episode_start_pos, dtype=float)
                goal_pos_arr = np.array(env.goal[:3], dtype=float)
                straight_distance = float(np.linalg.norm(start_pos_arr - goal_pos_arr))-2.0
            else:
                straight_distance = 0.0

            # 计算本episode的 RMS Jerk（基于加速度序列差分）
            if len(episode_accels) >= 2:
                accels = np.stack(episode_accels, axis=0)
                dt = float(mpc_controller.dt) if mpc_controller.dt > 0 else 1.0
                jerk_seq = (accels[1:] - accels[:-1]) / dt
                jerk_sq = np.sum(jerk_seq ** 2, axis=1)
                jerk_rms = float(np.sqrt(np.mean(jerk_sq)))
            else:
                jerk_rms = 0.0

            # 成功和失败的episode都要在Statistics中计数：
            # - 成功: 记录完整的距离、MPC统计和jerk等指标
            if success:
                statistics.record_episode(
                    success=success,
                    steps=step-1,
                    distance=episode_total_distance,
                    mpc_updates=episode_mpc_updates,
                    mpc_failures=episode_mpc_failures,
                    mpc_reuses=episode_mpc_reuses,
                    mpc_computations=episode_mpc_computations,
                    result_type=result_type,
                    straight_distance=straight_distance,
                    jerk_rms=jerk_rms,
                    trajectory=episode_trajectory,
                    velocities=episode_velocity,
                    accelerations=episode_acceleration,
                    trigger_seq=episode_trigger_seq,
                    dist_to_goal=episode_dist_to_goal,
                    q_weights=episode_q_weights,
                )
            else:
                statistics.record_episode(
                    success=success,
                    steps=0,
                    distance=0.0,
                    mpc_updates=0,
                    mpc_failures=0,
                    mpc_reuses=0,
                    mpc_computations=0,
                    result_type=result_type,
                    straight_distance=None,
                    jerk_rms=None,
                    trajectory=None,
                    velocities=None,
                    accelerations=None,
                    trigger_seq=None,
                    dist_to_goal=None,
                    q_weights=None,
                )
        
        
        # 根据train_policy参数选择训练哪个policy
        if not run_test_flag:
            if train_policy_q and policy_q is not None:
                # 训练Q权重调整policy
                if (policy_q.replayer.count > args.batch_size) and (epoch > 50):
                    policy_q.learn(velocity_rms=velocity_rms)
                if ((epoch != 0) and (epoch % 20 == 0)):
                    policy_q.save(epoch, policy_path + '/q')
                    logger.info('########################## Q权重Policy model saved when update {} times#########'
                                '################'.format(epoch))
            if train_policy_mpc and policy_mpc is not None:
                # 训练MPC更新控制policy
                if (policy_mpc.replayer.count > args.batch_size):
                    policy_mpc.learn(velocity_rms=velocity_rms)
                if ((epoch != 0) and (epoch % 20 == 0)):
                    policy_mpc.save(epoch, policy_path + '/mpc')
                    logger.info('########################## MPC更新控制Policy model saved when update {} times#########'
                                '################'.format(epoch))

        # 保留日志记录功能
        if not run_test_flag and writer is not None:
            writer.add_scalar("Train/average rewardq", np.sum(rewardq_list) / len(rewardq_list), epoch)
            writer.add_scalar("Train/average rewardmpc", np.sum(rewardmpc_list) / len(rewardmpc_list), epoch)
            writer.add_scalar("Train/rewardq", ep_rewardq, epoch)
            writer.add_scalar("Train/rewardmpc", ep_rewardmpc, epoch)
            writer.add_scalar("Train/success_rate", np.sum(sucess_list) / len(sucess_list), epoch)
            writer.flush()

    env.stop_polling()

    if not run_test_flag and writer is not None:
        writer.close()


if __name__ == '__main__':

    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)

    # =====================================================================
    # 【全局控制开关设置区】
    # 1. 训练与测试总开关 (两个都不训练即为测试模式)
    train_policy_q = False
    train_policy_mpc = False
    run_test_flag = not train_policy_q and not train_policy_mpc

    # 2. 模型加载总开关 (决定是否从本地文件读取历史模型)
    load_policy_q = True      
    load_policy_mpc = True    

    # 3. 参数配置
    map_index = 102
    policy_epoch_index_q = 1000
    policy_epoch_index_mpc = 1320
    load_replayer = False  # 是否加载经验回放池 (通常为 False)
    disp_info_flag = True # 是否在终端打印详细的步骤信息 (通常为 False)
    # =====================================================================

    if run_test_flag:
        kwargs_q['mode'] = 'test'
        kwargs_mpc['mode'] = 'test'
        args.max_episodes = 1
        print("测试模式：已设置最大episodes为{}".format(args.max_episodes))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Environment(rank, map_index, args.max_timesteps, args.dt)
    env.log_reward = args.log_reward
    policy_q = None
    policy_mpc = None
    statistics = None

    if run_test_flag:
        env.mode = 'test'
        if rank == 0:
            statistics = Statistics(capacity=100)
            print("测试模式：已创建Statistics实例用于统计指标")

    if rank == 0:
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = './log/' + hostname + '/output-' + date_time_str + '.log'
        logger = Logger(output_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
        
        # --- 创建 Policy 实例 ---
        if not run_test_flag:
            writer = SummaryWriter("./my_experiment/" + date_time_str)
            if train_policy_q or load_policy_q:
                policy_q = SAC_Ae_qMpc(env, writer=writer if train_policy_q else None, **kwargs_q)
                if not train_policy_q: 
                    policy_q.mode = 'test'
            if train_policy_mpc or load_policy_mpc:
                policy_mpc = SAC_Ae_eMpc(env, writer=writer if train_policy_mpc else None, **kwargs_mpc)
                if not train_policy_mpc: 
                    policy_mpc.mode = 'test'
        else:
            logger = Logger('./log/' + hostname + 'output.log', clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
            writer = None
            # 只创建需要加载的 policy 实例
            if load_policy_q:
                policy_q = SAC_Ae_qMpc(env, writer=None, **kwargs_q)
                policy_q.mode = 'test'
            if load_policy_mpc:
                policy_mpc = SAC_Ae_eMpc(env, writer=None, **kwargs_mpc)
                policy_mpc.mode = 'test'
            print(f"测试模式：policy_q={'已创建' if policy_q else '未创建'}, policy_mpc={'已创建' if policy_mpc else '未创建'}")

        # --- 设置路径 ---
        policy_path = './policy/DSAC'
        if not os.path.exists(policy_path): 
            os.makedirs(policy_path)
        q_policy_path = policy_path + '/q'
        mpc_policy_path = policy_path + '/mpc'
        if not os.path.exists(q_policy_path): 
            os.makedirs(q_policy_path)
        if not os.path.exists(mpc_policy_path): 
            os.makedirs(mpc_policy_path)
        q_model_path = policy_path + '/q'
        mpc_model_path = policy_path + '/mpc'

        # ====================================================================
        # 模型加载逻辑树 (严格按照 测试/训练 -> 加载 -> 视觉专属 的顺序)
        # ====================================================================
        starting_epoch_q = 0
        starting_epoch_mpc = 0

        if run_test_flag:
            # ---------------- 【A. 纯测试模式】 ----------------
            print("\n" + "="*40 + "\n进入纯测试模式加载流程\n" + "="*40)
            # 加载 Q权重 policy
            if load_policy_q and os.path.exists(q_model_path):
                print(f'>>> 加载完整的 Q权重policy 模型 (包含RL策略) from {q_model_path}')
                policy_q.load(q_model_path, 'test', policy_epoch_index_q)
            else:
                print(f'>>> [错误] 测试模式下 Q权重policy 模型不存在: {q_model_path}')
                
            # 加载 MPC更新控制 policy
            if load_policy_mpc and os.path.exists(mpc_model_path):
                print(f'>>> 加载完整的 MPC更新控制policy 模型 (包含RL策略) from {mpc_model_path}')
                policy_mpc.load(mpc_model_path, 'test', policy_epoch_index_mpc)
            else:
                print(f'>>> [错误] 测试模式下 MPC更新控制policy 模型不存在: {mpc_model_path}')         
        
        else:
            # ---------------- 【B. 训练及混合模式】 ----------------
            print("\n" + "="*40 + "\n进入训练/混合模式加载流程\n" + "="*40)
            
            for name, policy, train_flag, load_flag, model_path, epoch_idx in [
                ("Q权重", policy_q, train_policy_q, load_policy_q, q_model_path, policy_epoch_index_q),
                ("MPC控制", policy_mpc, train_policy_mpc, load_policy_mpc, mpc_model_path, policy_epoch_index_mpc),
            ]:
                if policy is None:
                    continue
                if train_flag:
                    if load_flag and os.path.exists(model_path):
                        print(f'>>> [{name}-断点续训] 加载完整历史模型继续训练 from {model_path}')
                        ep = policy.load(model_path, args.mode, epoch_idx)
                        if name == "Q权重":
                            starting_epoch_q = ep
                        else:
                            starting_epoch_mpc = ep
                    else:
                        print(f'>>> [{name}-从零训练] 模型未加载，网络完全随机初始化')
                else:
                    if load_flag and os.path.exists(model_path):
                        print(f'>>> [{name}-辅助推理] 加载完整的历史模型用于辅助测试 from {model_path}')
                        policy.load(model_path, 'test', epoch_idx)
                    else:
                        print(f'>>> [警告] {name}policy作为辅助但不加载模型，将使用随机参数')
    else:
        policy_q = None
        policy_mpc = None
        policy_path = None
        starting_epoch_q = 0
        starting_epoch_mpc = 0
        logger = None
        writer = None
        
    try:
        # 确定起始 epoch
        starting_epoch = max(starting_epoch_q, starting_epoch_mpc)
        if run_test_flag:
            starting_epoch = 0
            
        run(comm=comm, env=env, policy_q=policy_q, policy_mpc=policy_mpc, starting_epoch=starting_epoch,
            train_policy_q=train_policy_q, train_policy_mpc=train_policy_mpc, statistics=statistics)
        
        if run_test_flag and rank == 0 and statistics is not None:
            print("\n" + "="*80)
            print("测试完成！统计结果如下：")
            print("="*80)
            statistics.print_statistics(recent=False, verbose=True)            
            import json
            stats_dict = statistics.get_statistics_dict(recent=False)
            stats_filename = f'./log/{hostname}/sacmpc_statistics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            with open(stats_filename, 'w') as f:
                json.dump(stats_dict, f, indent=4)
            print(f"\n统计数据已保存到: {stats_filename}")

    except KeyboardInterrupt:
        if rank == 0:
            if not run_test_flag:
                if train_policy_q and policy_q is not None:
                    policy_q.writer.flush()
                    policy_q.writer.close()
                    policy_q.save(policy_q.epoch, policy_path + '/q', policy_q.goal_rms, policy_q.velocity_rms)
                    logger.info('########################## Q权重Policy model saved #################')
                if train_policy_mpc and policy_mpc is not None:
                    policy_mpc.writer.flush()
                    policy_mpc.writer.close()
                    policy_mpc.save(policy_mpc.epoch, policy_path + '/mpc', policy_mpc.goal_rms, policy_mpc.velocity_rms)
                    logger.info('########################## MPC更新控制Policy model saved #################')

import airsim
import numpy as np
import argparse
import json
from utils import Statistics, generate_points
import math
import logging
from Logger import Logger
import os
import time
import socket
from mpi4py import MPI
from datetime import datetime
from Environment import Environment
from Mpcstructure import MpcStructure
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 事件触发式 MPC (E-MPC) 不使用 Actor-Critic，自适应权重固定

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'empc.json')
with open(_CONFIG_PATH, 'r') as _f:
    _config = json.load(_f)

args = argparse.Namespace(
    policy=_config.get("policy", "E_MPC"),
    num_agent=_config.get("num_agent", 1),
    seed=_config.get("seed", 0),
    max_episodes=_config.get("max_episodes", 100),
    max_timesteps=_config.get("max_timesteps", 500),
    obs_shape=_config.get("obs_shape", [30]),
    dt=_config.get("dt", 0.2),
    mode=_config.get("mode", "test"),
    show_predicted_trajectory=_config.get("show_predicted_trajectory", True),
    fixed_points=_config.get("fixed_points", True),
    log_reward=_config.get("log_reward", True),
)

# 全局变量
disp_info_flag = False
logger = None
writer = None
policy_path = None


def convert_env_state_to_mpc_state(env):
    position = env.get_state(return_index=1)  # [x, y, z]
    state = env.client.getMultirotorState(vehicle_name="Drone_" + str(env.index))
    angular_vel = state.kinematics_estimated.angular_velocity
    local_goal, local_speed = env.get_local_goal_and_speed()
    quat = state.kinematics_estimated.orientation
    roll, pitch, yaw = airsim.to_eularian_angles(quat)
    
    # 显式获取全局 Z 轴线速度
    global_vz = state.kinematics_estimated.linear_velocity.z_val

    mpc_state = np.array([
        position[0],
        position[1],
        position[2],
        yaw,
        local_speed[0],        # vx (保持机体前向速度，这与MPC的 XY 模型匹配)
        angular_vel.z_val,     # wz
        global_vz,             # ⚠️ 将 local_speed[1] 替换为全局 global_vz
    ])
    return mpc_state, local_goal


def run(comm, env, starting_epoch=0, train_mode=False, statistics=None):
    """E-MPC 主循环：固定 Q，事件触发更新轨迹/MPC 求解"""
    success_list = []

    # 固定 MPC 代价权重（位置、控制、障碍物）
    fixed_Q = [0.5, 0.5, 3.0,   # 位置权重
               0.5, 0.5, 0.5,   # 控制平滑权重
               0.5]             # 障碍物权重

    horizon = 15
    max_velocity = 0.5
    max_acceleration = 1.0
    max_yaw_acceleration = 0.8
    max_yaw_rate = 0.8

    # 初始化 MPC 导航控制器
    mpc_controller = MpcStructure(
        Q=fixed_Q,
        dt=env.dt,
        horizon=horizon,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_yaw_acceleration=max_yaw_acceleration,
        max_yaw_rate=max_yaw_rate
    )
    mpc_controller.update_weights(weight=fixed_Q)
    env.get_obstacles()

    # 在测试模式下预先加载测试起点/终点数据
    test_points_data = None
    if env.index == 0:
        try:
            test_points_file = f'test_points{env.map_index}.npy'
            test_points_data = np.load(test_points_file)
            print(f"load {test_points_file}")
        except Exception as e:
            print(f"[Warning] Failed to load {test_points_file}: {e}. Fallback to random generate_points.")

    # 事件触发相关参数
    trigger_steps_before_end = 10     # 控制序列剩余 N 步以内触发

    for epoch in range(starting_epoch, args.max_episodes):
        env.client.simPause(False)
        terminal = False
        next_episode = False
        liveflag = True
        step = 1
        result = ''

        # 统计信息
        episode_mpc_updates = 0
        episode_mpc_failures = 0
        episode_mpc_reuses = 0
        episode_mpc_computations = 0
        episode_total_distance = 0.0
        episode_accels = []

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

        if env.index == 0:
            if test_points_data is not None:
                total_points = test_points_data.shape[0]
                if total_points < args.num_agent:
                    raise ValueError("test_points.npy 中的数据数量小于 num_agent")
                max_start = total_points - args.num_agent
                base_idx = (epoch * args.num_agent) % (max_start + 1)
                selected = test_points_data[base_idx:base_idx + args.num_agent]
                pose_list = selected[:, 0, :].tolist()
                goal_list = selected[:, 1, :].tolist()
            else:
                pose_list, goal_list = generate_points(num_env=args.num_agent, map_index=env.map_index, fixed=args.fixed_points)
        else:
            pose_list, goal_list = None, None

        # 统一物理引擎重置逻辑
        env.reset_world()
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]
        
        temp_pose = airsim.Pose()
        temp_pose.position.x_val = pose_ctrl[0]
        temp_pose.position.y_val = pose_ctrl[1]
        temp_pose.position.z_val = pose_ctrl[2]
        env.client.simSetVehiclePose(temp_pose, ignore_collision=True, vehicle_name="Drone_" + str(env.index))
        
        time.sleep(2)
        env.drones_init()
        time.sleep(1)
        
        init_pose = env.get_state()
        env.generate_goal_point(goal_ctrl)
        env.reset_pose(init_pose, pose_ctrl)

        current_mpc_state, local_goal = convert_env_state_to_mpc_state(env)
        episode_start_pos = env.get_state(return_index=1)
        last_pos_array = np.array(episode_start_pos, dtype=np.float32)
        last_vel = np.zeros(3, dtype=np.float32)

        # 控制序列缓存
        cached_control_sequence = None
        cached_step_index = 0 

        while not next_episode:
            if liveflag:
                step_start_time = time.time()
                env.get_obstacle_vector()

                current_mpc_state, _ = convert_env_state_to_mpc_state(env)
                current_pos = current_mpc_state[:3]

                # ================= 1. 纯视觉事件触发条件 =================
                need_replan = False

                # 获取当前到目标的相对角度误差
                current_pos = current_mpc_state[:3]
                global_goal = np.array(env.goal)
                pos_error = global_goal - current_pos

                target_yaw = np.arctan2(pos_error[1], pos_error[0])
                current_yaw = current_mpc_state[3]
                yaw_error = abs(np.arctan2(np.sin(target_yaw - current_yaw), 
                                           np.cos(target_yaw - current_yaw)))

                # 条件1：视觉感知触发 (FOV内有障碍物)
                if len(env.obstacles) > 0:
                    need_replan = True  # 看到障碍物，立即触发重规划
                
                # 条件2：姿态审核与曲率拒绝 (姿态不对或轨迹不对)
                if yaw_error > math.radians(20.0):
                    need_replan = True

                # 条件3：序列耗尽兜底触发
                if cached_control_sequence is None:
                    need_replan = True
                else:
                    remaining_steps = len(cached_control_sequence) - cached_step_index
                    if remaining_steps <= trigger_steps_before_end:
                        need_replan = True

                # 条件4：速度过低时强制触发，防止无人机停滞
                current_speed = np.sqrt(current_mpc_state[4]**2 + current_mpc_state[6]**2)
                if current_speed < 0.3:
                    need_replan = True

                # 求解 MPC
                replan_succeeded = False
                if need_replan:
                    try:
                        episode_mpc_computations += 1
                        global_goal = np.array(env.goal)
                        mpc_solve_start = time.time()
                        cached_control_sequence, success = mpc_controller.get_velocity_command(
                            current_mpc_state, global_goal, obstacles=env.obstacles
                        )
                        mpc_solve_time = time.time() - mpc_solve_start 
                        print(f"Step {step}: MPC 求解耗时 {mpc_solve_time:.4f} 秒")

                        if success and cached_control_sequence is not None:
                            episode_mpc_updates += 1
                            cached_step_index = 0
                            replan_succeeded = True
                        else:
                            cached_control_sequence = None
                            episode_mpc_failures += 1

                    except Exception:
                        episode_mpc_computations += 1
                        episode_mpc_failures += 1
                        cached_control_sequence = None

                # 执行控制
                if cached_control_sequence is not None and cached_step_index < len(cached_control_sequence):
                    mpc_velocity = cached_control_sequence[cached_step_index]
                    cached_step_index += 1
                    if not replan_succeeded:          # 只有不是刚刚求解的才算复用
                        episode_mpc_reuses += 1
                else:
                    # 使用与 onlympc 完全一致的 fallback
                    current_pos = current_mpc_state[:3]
                    global_goal = np.array(env.goal)
                    pos_error = global_goal - current_pos
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
                    if abs(yaw_error) > np.pi / 4:
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

                # 积分为速度指令
                current_vx = current_mpc_state[4]
                current_wz = current_mpc_state[5]
                current_vz = current_mpc_state[6]

                dt = mpc_controller.dt
                new_vx = current_vx + mpc_velocity[0] * dt
                new_wz = current_wz + mpc_velocity[1] * dt
                new_vz = current_vz + mpc_velocity[2] * dt

                episode_accels.append(np.array(mpc_velocity, dtype=float))

                velocity_magnitude = np.sqrt(new_vx ** 2 + new_vz ** 2)
                if velocity_magnitude > mpc_controller.max_velocity:
                    scale = mpc_controller.max_velocity / velocity_magnitude
                    new_vx *= scale
                    new_vz *= scale

                if abs(new_wz) > mpc_controller.max_yaw_rate:
                    new_wz = np.sign(new_wz) * mpc_controller.max_yaw_rate

                velocity = [new_vx, new_wz, new_vz]

                if args.show_predicted_trajectory and cached_control_sequence is not None:
                    remaining_steps = len(cached_control_sequence) - cached_step_index
                    if remaining_steps > 0:
                        plot_sequence = np.zeros_like(cached_control_sequence)
                        plot_sequence[:remaining_steps] = cached_control_sequence[cached_step_index:]
                        predicted_trajectory = mpc_controller.predict_trajectory(
                            current_mpc_state, plot_sequence
                        )
                        
                        # 4. 只绘制有效的剩余轨迹段
                        env.plot_predicted_trajectory(
                            predicted_trajectory[:remaining_steps + 1],
                            color_rgba=[0.0, 1.0, 0.0, 1.0],  # 绿色
                            thickness=8,
                            duration=0.2,  # 与 dt 保持一致，防止频闪
                        )

                # 发送控制指令给底层
                # 与环境和交互
                env.control_vel(velocity)

                is_crashed = env.get_crash_state()

                # 统一环境判定接口
                _, _, terminal, result = env.get_reward_terminate_result(step, update_flag=need_replan, is_crashed=is_crashed)
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
                episode_trigger_seq.append(1 if need_replan else 0)
                episode_dist_to_goal.append(float(np.linalg.norm(current_pos_array - np.array(env.goal[:3]))))
                episode_q_weights.append([float(v) for v in fixed_Q])

                elapsed_time = time.time() - step_start_time
                if elapsed_time < args.dt:
                    time.sleep(args.dt - elapsed_time)

                step += 1
                env.step = step

            if terminal:
                liveflag = False
                next_episode = True
                if result == 'Reach Goal':
                    if len(success_list) == 200:
                        success_list.pop(0)
                    success_list.append(1)
                else:
                    if len(success_list) == 200:
                        success_list.pop(0)
                    success_list.append(0)

        # 一个 episode 结束
        if logger is not None:
            logger.info('E-MPC Env %02d, Goal (%05.1f, %05.1f, %05.1f), Episode %05d, step %03d, '
                        '%s, map%d, success_rate:%02.2f, mpc_computations:%d, mpc_reuses:%d, mpc_failures:%d' %
                        (env.index, env.goal[0], env.goal[1], env.goal[2], epoch + 1, step,
                         result, env.map_index, np.sum(success_list) / len(success_list) if success_list else 0,
                         episode_mpc_computations, episode_mpc_reuses, episode_mpc_failures))

        env.plot_path()
        if disp_info_flag:
            print('[E-MPC] Result %s' % (result))
        env.client.simPause(False)

        # 统计记录 (与 onlympc 对齐)
        if statistics is not None:
            success = (result == 'Reach Goal')
            if 'collision' in result.lower() or 'crash' in result.lower():
                result_type = 'collision'
            elif 'timeout' in result.lower() or 'time' in result.lower():
                result_type = 'timeout'
            elif success:
                result_type = 'success'
            else:
                result_type = 'other'

            # 计算直线距离
            if episode_start_pos is not None and hasattr(env, "goal") and len(env.goal) >= 3:
                start_pos_arr = np.array(episode_start_pos, dtype=float)
                goal_pos_arr = np.array(env.goal[:3], dtype=float)
                straight_distance = float(np.linalg.norm(goal_pos_arr - start_pos_arr)) - 2.0
            else:
                straight_distance = 0.0

            # 计算 Jerk
            if len(episode_accels) >= 2:
                accels = np.stack(episode_accels, axis=0)
                dt = float(mpc_controller.dt) if mpc_controller.dt > 0 else 1.0
                jerk_seq = (accels[1:] - accels[:-1]) / dt
                jerk_sq = np.sum(jerk_seq ** 2, axis=1)
                jerk_rms = float(np.sqrt(np.mean(jerk_sq)))
            else:
                jerk_rms = 0.0

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

if __name__ == '__main__':
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)

    starting_epoch = 0
    map_index = 102
    disp_info_flag = True
    train_mode = False  

    args.max_episodes = 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Environment(rank, map_index, args.max_timesteps, args.dt)
    env.log_reward = args.log_reward
    statistics = None

    env.mode = 'test'
    if rank == 0:
        statistics = Statistics(capacity=100)

    if rank == 0:
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = './log/' + hostname + '/output-empc-' + date_time_str + '.log'
        logger = Logger(output_file, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
        print("E-MPC initialized successfully")
    else:
        logger = None
        writer = None

    try:
        run(comm=comm, env=env, starting_epoch=starting_epoch,
            train_mode=train_mode, statistics=statistics)

        if rank == 0 and statistics is not None:
            print("\n" + "=" * 80)
            print("E-MPC 测试完成！统计结果如下：")
            print("=" * 80)
            statistics.print_statistics(recent=False, verbose=True)            
            
            import json
            stats_dict = statistics.get_statistics_dict(recent=False)
            stats_filename = f'./log/{hostname}/empc_statistics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            with open(stats_filename, 'w') as f:
                json.dump(stats_dict, f, indent=4)
            print(f"\n统计数据已保存到: {stats_filename}")
    except Exception as e:
        if rank == 0:
            print(f"[E-MPC] Exception: {e}")
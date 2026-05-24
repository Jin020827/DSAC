import airsim
import numpy as np
import argparse
import os
import time
import socket
from mpi4py import MPI
from datetime import datetime

from Environment import Environment
from Mpcstructure import MpcStructure
from utils import Statistics, generate_points


import json

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'onlympc.json')
with open(_CONFIG_PATH, 'r') as _f:
    _config = json.load(_f)

args = argparse.Namespace(
    num_agent=_config.get("num_agent", 1),
    num_barrier=_config.get("num_barrier", 0),
    seed=_config.get("seed", 0),
    max_episodes=_config.get("max_episodes", 200),
    max_timesteps=_config.get("max_timesteps", 500),
    obs_shape=_config.get("obs_shape", [30]),
    dt=_config.get("dt", 0.2),
    show_predicted_trajectory=_config.get("show_predicted_trajectory", True),
    fixed_points=_config.get("fixed_points", True),
    log_reward=_config.get("log_reward", True),
)

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


def run_only_mpc(comm, env, statistics=None, starting_epoch=0):
    crash_count = 0
    sucess_list = []

    # 固定 MPC 代价权重（位置、控制、障碍物）
    # fixed_Q = [0.5, 0.5, 3.0,   # 位置权重
    #            0.5, 0.5, 0.5,   # 控制平滑权重
    #            1.0]             # 障碍物权重
    fixed_Q = [0.5, 0.5, 3.0,   # 位置权重
               0.5, 0.5, 0.5,   # 控制平滑权重
               0.5]             # 障碍物权重

    horizon = 15
    max_velocity = 0.5
    max_acceleration = 1.0
    max_yaw_acceleration = 1.0
    max_yaw_rate = 1.0

    # 初始化MPC控制器
    mpc_controller = MpcStructure(
        Q=fixed_Q,
        dt=env.dt,  # 采样时间
        horizon=horizon,  # 预测时域长度
        max_velocity=max_velocity,  # 最大速度限制
        max_acceleration=max_acceleration,  # 最大加速度限制
        max_yaw_acceleration=max_yaw_acceleration,  # 最大角加速度限制
        max_yaw_rate=max_yaw_rate  # 最大角速度限制
    )

    mpc_controller.update_weights(weight=fixed_Q)
    env.get_obstacles()

    last_mpc_control_sequence = None
    mpc_step_counter = 0

    # 预先加载测试起点/终点数据
    test_points_data = None
    if env.index == 0:
        try:
            test_points_file = f'test_points{env.map_index}.npy'
            test_points_data = np.load(test_points_file)
            print(f"load {test_points_file}")
        except Exception as e:
            print(f"[Warning] Failed to load {test_points_file}: {e}. Fallback to random generate_points.")

    env.start_polling(interval=0.02)  # 50Hz 轮询

    for epoch in range(starting_epoch, args.max_episodes):
        env.client.simPause(False)
        terminal = False
        next_episode = False
        liveflag = True
        step = 1
        result = ""

        last_mpc_control_sequence = None
        mpc_step_counter = 0

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
        episode_trigger_seq = []   # 每步MPC触发信号（onlympc每步都触发）[0/1, ...]
        episode_dist_to_goal = []  # 每步到目标距离 [d, ...]
        episode_q_weights = []     # 每步Q权重 [[q0..q6], ...]
        all_episodes_mpc_time = []
        episode_total_mpc_time = 0.0
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
         
        env.reset_world()
        pose_ctrl = pose_list[env.index]
        goal_ctrl = goal_list[env.index]
        
        # 临时构建一个 Pose 对象用于瞬移
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


        while not next_episode:
            if liveflag:
                step_start_time = time.time()
                env.move_obstacles(step)
                try:
                    # 每步都重新进行一次 MPC 优化
                    env.get_obstacle_vector()
                    current_mpc_state, _ = convert_env_state_to_mpc_state(env)
                    global_goal = np.array(env.goal)

                    episode_mpc_computations += 1
                    mpc_solve_start = time.time()
                    mpc_control_sequence, success = mpc_controller.get_velocity_command(
                        current_mpc_state, global_goal, obstacles=env.obstacles
                    )
                    mpc_solve_time = time.time() - mpc_solve_start
                    # print(f"Step {step}: MPC 求解耗时 {mpc_solve_time:.4f} 秒")
                    episode_total_mpc_time += mpc_solve_time

                    if success and mpc_control_sequence is not None:
                        episode_mpc_updates += 1
                        last_mpc_control_sequence = mpc_control_sequence
                        last_mpc_index = 0
                        mpc_velocity = mpc_control_sequence[0]
                    else:
                        episode_mpc_failures += 1
                        # fallback：简单的P控制朝向目标
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
                        az_desired = np.clip((desired_vz - current_vz) / mpc_controller.dt, -0.5, 0.5)
                        aw_desired = desired_wz
                        mpc_velocity = np.array([ax_desired, aw_desired, az_desired])

                    if last_mpc_control_sequence is not None:
                        current_mpc_state, _ = convert_env_state_to_mpc_state(env)
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
                    else:
                        velocity = [0.0, 0.0, 0.0]

                    mpc_step_counter += 1
                except Exception as e:
                    # 异常情况下也计入一次MPC计算和失败
                    episode_mpc_computations += 1
                    episode_mpc_failures += 1
                    if 'velocity' not in locals():
                        velocity = [0.0, 0.0, 0.0]
                    print(f"Exception caught: {e}")

                if args.show_predicted_trajectory and last_mpc_control_sequence is not None:
                    predicted_trajectory = mpc_controller.predict_trajectory(
                        current_mpc_state, last_mpc_control_sequence
                    )
                    env.plot_predicted_trajectory(
                        predicted_trajectory,
                        color_rgba=[0.0, 1.0, 0.0, 1.0],
                        thickness=8,
                        duration=0.1,
                    )

                # env.client.simPause(False)
                
                # 与环境和交互
                env.control_vel(velocity)
                if args.show_predicted_trajectory:
                    env.client.simFlushPersistentMarkers()
                # env.client.simPause(True)
                
                is_crashed = env.get_crash_state()

                # 
                _, _, terminal, result = env.get_reward_terminate_result(step, update_flag=True, is_crashed=is_crashed)
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
                episode_trigger_seq.append(1)  # onlympc 每步都触发
                episode_dist_to_goal.append(float(np.linalg.norm(current_pos_array - np.array(env.goal[:3]))))
                episode_q_weights.append([float(v) for v in fixed_Q])

                current_mpc_state, _ = convert_env_state_to_mpc_state(env)

                elapsed_time = time.time() - step_start_time
                if elapsed_time < args.dt:
                    time.sleep(args.dt - elapsed_time)

                step += 1
                env.step = step

            if terminal:
                liveflag = False
                next_episode = True
                if result == "Reach Goal":
                    if len(sucess_list) == 200:
                        sucess_list.pop(0)
                    sucess_list.append(1)
                else:
                    if len(sucess_list) == 200:
                        sucess_list.pop(0)
                    sucess_list.append(0)

        print(f"当前回合总用时：{episode_total_mpc_time}")
        
        env.plot_path()
        env.client.simPause(False)
        print("Episode {} finished. Result: {}".format(epoch, result))

        if statistics is not None:
            success = result == "Reach Goal"
            if "collision" in result.lower() or "crash" in result.lower():
                result_type = "collision"
            elif "timeout" in result.lower() or "time" in result.lower():
                result_type = "timeout"
            elif success:
                result_type = "success"
            else:
                result_type = "other"

            if episode_start_pos is not None and hasattr(env, "goal") and len(env.goal) >= 3:
                start_pos_arr = np.array(episode_start_pos, dtype=float)
                goal_pos_arr = np.array(env.goal[:3], dtype=float)
                straight_distance = float(np.linalg.norm(start_pos_arr - goal_pos_arr)) - 2.0
            else:
                straight_distance = 0.0

            if len(episode_accels) >= 2:
                accels = np.stack(episode_accels, axis=0)
                dt = float(mpc_controller.dt) if mpc_controller.dt > 0 else 1.0
                jerk_seq = (accels[1:] - accels[:-1]) / dt
                jerk_sq = np.sum(jerk_seq ** 2, axis=1)
                jerk_rms = float(np.sqrt(np.mean(jerk_sq)))
            else:
                jerk_rms = 0.0

            # 成功和失败的 episode 在统计上的处理与 Drone_mpc.py 保持一致：
            # - 成功: 记录完整的距离、MPC 统计和 jerk 等指标
            # - 失败: 只记录失败次数等基本信息，其余指标置零/None
            # 计算该回合的平均偏航角速度
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

    env.stop_polling()


if __name__ == "__main__":
    hostname = socket.gethostname()
    if not os.path.exists("./log/" + hostname):
        os.makedirs("./log/" + hostname)

    map_index = 102

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Environment(rank, map_index, args.max_timesteps, args.dt)
    env.log_reward = args.log_reward
    statistics = None

    env.mode = "test"
    if rank == 0:
        statistics = Statistics(capacity=100)
    
    args.max_episodes = 1

    starting_epoch = 0
    run_only_mpc(comm=comm, env=env, statistics=statistics, starting_epoch=starting_epoch)

    if rank == 0 and statistics is not None:
        print("\n" + "=" * 80)
        print("传统MPC测试完成！统计结果如下：")
        print("=" * 80)
        statistics.print_statistics(recent=False, verbose=True)
        # 可选：保存统计数据为JSON
        import json
        stats_dict = statistics.get_statistics_dict(recent=False)
        stats_filename = f'./log/{hostname}/onlympc_statistics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
        with open(stats_filename, 'w') as f:
            json.dump(stats_dict, f, indent=4)
            print(f"\n统计数据已保存到: {stats_filename}")
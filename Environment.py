import airsim
import numpy as np
from utils import global2body
import time
import math
import copy
from queue import Queue
import random
import threading
import os
import subprocess
from Logger import Logger
import logging
from datetime import datetime


def resolve_airsim_ip():
    configured_ip = os.environ.get("AIRSIM_IP")
    if configured_ip:
        return configured_ip

    if os.environ.get("WSL_DISTRO_NAME"):
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            route_parts = result.stdout.split()
            if "via" in route_parts:
                return route_parts[route_parts.index("via") + 1]
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    return "127.0.0.1"



class Environment():
    def __init__(self, index, map_index, max_timesteps, dt, log_reward=True):
        # connect to the AirSim simulator
        self.airsim_ip = resolve_airsim_ip()
        print(f"Connecting to AirSim at {self.airsim_ip}:41451")
        self.client = airsim.MultirotorClient(ip=self.airsim_ip)
        self.client.confirmConnection()
        self.index = index
        self.max_timesteps = max_timesteps
        self.reset_flag = False
        self.hover_flag = False
        self.record_pos = []
        self.obstacle_list = []
        self.stop_count = 0
        self.vel = [0, 0, 0]
        self.map_index = map_index
        self.mode = 'train'
        self.goal = [10, 10]
        self.dt = dt
        self.log_reward = log_reward
        self.obstacles = []         # 当前帧对无人机“有效”的障碍物
        self.all_obstacles = []     # 地图中“所有”的障碍物（真值地图）
        self.dynamic_origin_cache = {} # 用于锁定动态障碍物的震动中心
        self._dynamic_positions = {}   # 内存中追踪的动态障碍物当前位置 {obj_name: np.array([x, y, z])}
        self.pre_distance = 0
        self.distance = 0
        self.initial_distance = 0
        self.pre_control = [0, 0, 0]  # 保存前一个时间步的控制输入 [Vx, Vw, Vz]
        self.current_control = [0, 0, 0]  # 当前时间步的控制输入 [Vx, Vw, Vz]
        self.step = 0
        self.current_epoch = 0

        # 多线程相关
        # 后台状态缓存
        self._state_cache = {
            'pos': [0.0, 0.0, 0.0],
            'has_collided': False,
            'timestamp': 0.0
        }
        self._state_lock = threading.Lock()
        self._polling_active = False
        self._polling_thread = None

        # 奖励日志
        log_dir = './log'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'reward_drone_{self.index}_{timestamp}.log')
        self.reward_logger = Logger(
            log_path,
            clevel=logging.DEBUG,
            Flevel=logging.DEBUG,
            CMD_render=False,
            propagate = False
        )
        # 写入表头
        self.reward_logger.info(
            'epoch, ' # 回合
            'step, ' # 步数
            'result, ' # 结果
            'dist_curr, ' # 距离
            'num_obstacles, ' # 障碍物数量
            'yaw_err, ' # 偏航角
            'q, ' # 当前权重
            'r_weight, ' # 权重奖励
            'r1, ' # 总奖励
            'r_safe,' # 有障碍物奖励
            'r_yaw,' # 无障碍物奖励
            'r2,' # 总奖励 
            'risk_relu_list'
        )

    def reset_world(self):
        if self.index == 0:
            self.client.reset()
        # 重置平滑性奖励相关的控制输入历史
        self.pre_control = [0, 0, 0]  # 重置前一个控制输入
        self.current_control = [0, 0, 0]  # 重置当前控制输入

        self.all_obstacles = [] 
        self.obstacles = []
        self.step = 0

        # 重置碰撞缓存，防止上一 episode 的碰撞状态污染下一 episode
        with self._state_lock:
            self._state_cache['has_collided'] = False

        # 将动态障碍物归位到初始位置（内存 + 仿真器同步）
        for obj_name, init_pos in self.dynamic_origin_cache.items():
            self._dynamic_positions[obj_name] = init_pos.copy()
            try:
                # 读取原始姿态（仅保留旋转），构造新 Pose 写回
                orig_state = self.client.simGetObjectPose(object_name=obj_name)
                reset_pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        float(init_pos[0]),
                        float(init_pos[1]),
                        float(init_pos[2])
                    ),
                    orientation_val=orig_state.orientation
                )
                self.client.simSetObjectPose(object_name=obj_name, pose=reset_pose, teleport=True)
            except Exception as e:
                print(f"[reset_world] Failed to reset {obj_name}: {e}")

        # print(self.get_state())
        # time.sleep(2)
        # print(self.get_state())
        # # 设置无人机的新位置和姿态
        # new_position = airsim.Vector3r(x_val=0, y_val=0, z_val=-20)  # 新位置坐标 (x, y, z)
        # new_orientation = airsim.to_quaternion(pitch=0, roll=0, yaw=0)  # 新方向，这里设置为偏航90度
        # # 创建Pose对象
        # pose = airsim.Pose(new_position, new_orientation)
        # # 设置无人机的位姿
        # self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=f"{'Drone_' + str(self.index)}")
        # print(self.get_state())

    def drones_init(self):
        # 清除绘制痕迹
        self.client.simFlushPersistentMarkers()
        self.record_pos = []
        self.client.enableApiControl(True, vehicle_name="Drone_" + str(self.index))
        self.client.armDisarm(True, vehicle_name="Drone_" + str(self.index))

    def reset_pose(self, init_pos, target_pos):
        """
            重置无人机位姿到指定起点，并强制朝向目标点。
            
            Args:
                init_pos: 当前状态 (无用，保留接口兼容)
                target_pos: [x, y, z] 无人机要去的起点坐标
        """
        assert len(target_pos) == 3 or len(target_pos) == 4
        Pose = airsim.Pose()
        
        # 1. 设置位置
        # 注意：这里 target_pos 其实是 generate_points 生成的 start_point
        pose = list(np.array(target_pos[:3]))
        [Pose.position.x_val, Pose.position.y_val, Pose.position.z_val] = pose

        # 2. 设置朝向
        if len(target_pos) == 3:
            # 计算从 起点(target_pos) 指向 终点(self.goal) 的向量
            dx = self.goal[0] - target_pos[0]
            dy = self.goal[1] - target_pos[1]
            
            # 计算目标偏航角 (Yaw)
            initial_yaw = math.atan2(dy, dx)
            
            # 将欧拉角转换为四元数 (Roll=0, Pitch=0, Yaw=initial_yaw)
            qtn = airsim.to_quaternion(0, 0, initial_yaw)
            
            # 打印调试日志，确认朝向已设置
            # print(f" [Reset Pose] Start: {target_pos[:2]}, Goal: {self.goal[:2]}, Yaw: {math.degrees(initial_yaw):.1f}°")
        else:
            # 如果输入包含第4维，说明外部指定了朝向 (保留旧逻辑)
            qtn = airsim.to_quaternion(0, 0, target_pos[3])
            
        [Pose.orientation.x_val, Pose.orientation.y_val, Pose.orientation.z_val, Pose.orientation.w_val] = qtn
        
        # 3. 执行瞬移
        self.client.simSetVehiclePose(vehicle_name="Drone_" + str(self.index), ignore_collision=False, pose=Pose)
        
        # 4. 稳定物理状态
        # 瞬移后物理引擎会有惯性抖动，稍微悬停一下让状态收敛
        time.sleep(0.5) 
        
        # 5. 确保高度锁定
        # 强制将 Z 轴拉回到设定高度，防止瞬移导致的掉高
        # self.client.moveToZAsync(target_pos[2], velocity=2, vehicle_name="Drone_" + str(self.index), timeout_sec=5).join()
        
        # 再次等待稳定
        time.sleep(0.2)

    def get_state(self, return_index=1):
        pose = self.client.simGetVehiclePose(vehicle_name='Drone_' + str(self.index))
        position = pose.position
        quaternion = pose.orientation
        x, y, z = position
        w, w_x, w_y, w_z = quaternion
        if return_index == 1:
            return [x, y, z]
        elif return_index == 2:
            return [w, w_x, w_y, w_z]
        elif return_index == 3:
            return position
        elif return_index == 4:
            return quaternion
        elif return_index == 5:
            orientation = pose.orientation
            pitch, roll, yaw = airsim.to_eularian_angles(orientation)
            yaw_degrees = math.degrees(yaw)
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)

            return [x, y, z, yaw_degrees, sin_yaw, cos_yaw]
        elif return_index == 6:
            current_pos = np.array(self.get_state(return_index=1))
            dist_curr = np.linalg.norm(current_pos - np.array(self.goal))

            # 航向角误差（机体坐标系）
            local_goal_vec, _ = self.get_local_goal_and_speed()
            yaw = abs(math.atan2(local_goal_vec[1], local_goal_vec[0]))
            # yaw_err ∈ [0, π]，归一化到 [0, 1] 后取负作为惩罚
            yaw_err = -(yaw / math.pi)   # 正对目标时为 0，完全反向时为 -1
            return yaw_err


        return None

    def get_local_goal_and_speed(self):
        state = self.client.getMultirotorState(vehicle_name="Drone_" + str(self.index))
        Quaternious = state.kinematics_estimated.orientation
        [roll, pitch, yaw] = airsim.to_eularian_angles(Quaternious)
        GT_goal = np.asarray(self.goal)
        pos = self.get_state()
        pos = np.asarray(pos)
        local_goal = global2body(roll, pitch, yaw, GT_goal, pos)
        v_xyz = np.array(
            [state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val,
             state.kinematics_estimated.linear_velocity.z_val])
        v_xyz = global2body(roll, pitch, yaw, v_xyz, np.array([0, 0, 0]))
        vx, vz = v_xyz[0], v_xyz[2]
        vw = state.kinematics_estimated.angular_velocity.z_val
        local_speed = np.asarray([vx, vz, vw])

        return local_goal, local_speed

    def start_polling(self, interval=0.01):
        """启动后台轮询线程，interval 单位秒，默认 50Hz"""
        if self._polling_thread is not None and self._polling_thread.is_alive():
            return
        self._polling_active = True
        self._polling_thread = threading.Thread(
            target=self._poll_state,
            args=(interval,),
            daemon=True  # 主线程退出时自动结束
        )
        self._polling_thread.start()

    def stop_polling(self):
        """停止后台轮询线程"""
        self._polling_active = False
        if self._polling_thread is not None:
            self._polling_thread.join(timeout=1.0)
            self._polling_thread = None

    def _check_physical_collision(self, drone_pos):
        """
        物理碰撞检测：将障碍物视为无限高的圆柱体。
        仅对静态障碍物计算水平距离；动态障碍物依赖仿真底层碰撞检测。
        """
        R_DRONE = 0.4          # 无人机物理半径
        SAFETY_OFFSET = 0.1    # 安全余量
        
        check_list = self.all_obstacles
        if not check_list: 
            return False, None

        # 提取无人机在 XY 平面的投影坐标
        drone_pos_xy = drone_pos[0:2]

        for obs in check_list:
            # === 核心修改：如果是动态障碍物，直接跳过数学距离检测 ===
            if obs.get("type") == "dynamic":
                continue
                
            # 提取障碍物中心在 XY 平面的投影坐标
            obs_pos_xy = obs["center"][0:2]
            # 获取该障碍物的物理半径（Map 102 中默认为 0.5）
            r_obs = obs.get("radius", 0.5) 
            
            # 计算水平面上的欧氏距离（即到圆柱中心轴的距离）
            dist_xy = np.linalg.norm(drone_pos_xy - obs_pos_xy)
            
            # 判定条件：水平距离 < (无人机半径 + 障碍物半径 + 余量)
            if dist_xy < (R_DRONE + r_obs + SAFETY_OFFSET):
                # 如果进入了圆柱体范围内，立即触发碰撞
                return True, f"Static Obstacle {obs.get('name', 'Unknown')} (Horizontal Dist: {dist_xy:.2f})"
                
        return False, None
    
    def _poll_state(self, interval=0.01):
        poll_client = airsim.MultirotorClient(ip=self.airsim_ip)
        poll_client.confirmConnection()

        while self._polling_active:
            try:
                # 高频从仿真软件拉取最新位置
                pose = poll_client.simGetVehiclePose(
                    vehicle_name='Drone_' + str(self.index)
                )
                pos = [pose.position.x_val,
                    pose.position.y_val,
                    pose.position.z_val]

                # 高频从仿真软件拉取最新碰撞状态
                is_sim_crash = False
                if self.step >= 2:
                    crash_info = poll_client.simGetCollisionInfo(
                        vehicle_name="Drone_" + str(self.index)
                    )
                    is_sim_crash = crash_info.has_collided

                # 写入缓存（加锁保护）
                with self._state_lock:
                    self._state_cache['pos'] = pos
                    self._state_cache['has_collided'] = is_sim_crash
                    self._state_cache['timestamp'] = time.time()

            except Exception:
                pass

            time.sleep(interval)

    def get_crash_state(self):
        # 主线程读缓存（副线程已经拉取了最新状态）
        with self._state_lock:
            pos = self._state_cache['pos']
            is_sim_crash = self._state_cache['has_collided']

        # 用缓存里的最新位置做物理碰撞检测（计算量小，主线程自己算）
        drone_pos = np.array(pos)
        is_phy_crash, _ = self._check_physical_collision(drone_pos)

        return is_phy_crash or is_sim_crash

    def get_obstacle_vector(self):
        MAX_DIST = 6.0
        FOV_COS_THRESHOLD = 0.707  # cos(45°)
        max_obs = 5
        feature_dim = 6
        final_vector = np.zeros(max_obs * feature_dim, dtype=np.float32)

        if len(self.all_obstacles) == 0:
            self.get_obstacles()
        
        # 1. 获取并检查无人机状态 (第一道防线)
        uav_state = self.get_state(return_index=5)  # [x, y, z, yaw_deg]
        if uav_state is None or np.isnan(uav_state).any():
            return final_vector # 自身状态坏死，直接返回全0安全向量

        uav_pos = np.array(uav_state[:3])
        uav_yaw_rad = math.radians(uav_state[3])
        forward_vec = np.array([math.cos(uav_yaw_rad), math.sin(uav_yaw_rad)])

        # 获取机体姿态（用于坐标转换）
        pose = self.client.simGetVehiclePose(vehicle_name="Drone_" + str(self.index))
        # 确保四元数不是 NaN
        if math.isnan(pose.orientation.w_val):
            return final_vector
            
        roll, pitch, yaw = airsim.to_eularian_angles(pose.orientation)

        obs_list = []
        for obstacle in self.all_obstacles:
            obs_center = obstacle["center"].copy()
            
            # 2. 检查障碍物中心坐标 (第二道防线)
            if np.isnan(obs_center).any():
                continue

            obs_center[2] = uav_pos[2]  # Z对齐无人机高度

            rel_pos_global = obs_center[:2] - uav_pos[:2]
            dist = np.linalg.norm(rel_pos_global)

            # 3. 检查计算出的距离是否合法 (第三道防线)
            if np.isnan(dist) or dist < 1e-3:
                continue

            # 距离筛选（固定6m）
            if dist >= MAX_DIST:
                continue

            # 视野角度筛选（±45度）
            rel_pos_unit = rel_pos_global / dist
            dot_product = np.dot(forward_vec, rel_pos_unit)
            if np.isnan(dot_product) or dot_product <= FOV_COS_THRESHOLD:
                continue

            # 转换到机体坐标系
            local_pos = global2body(roll, pitch, yaw, obs_center, uav_pos)

            # 归一化距离（基于6m视野）
            norm_dist = np.clip(dist / MAX_DIST, 0.0, 1.0)

            # 偏航角
            angle = math.atan2(local_pos[1], local_pos[0])
            obs_feature = [
                local_pos[0], local_pos[1], local_pos[2],
                norm_dist,
                math.sin(angle), math.cos(angle)
            ]
            
            # 4. 最后确保没有把 NaN 塞进去 (最后一道防线)
            if np.isnan(obs_feature).any():
                continue
                
            obs_list.append((dist, obs_feature, obstacle)) 

        # 按距离从小到大排序（近的优先）
        obs_list.sort(key=lambda x: x[0])

        # 更新 self.obstacles（供MPC使用）
        self.obstacles = [item[2] for item in obs_list]
        # print(f"看到的障碍物：{self.obstacles}")

        # 填充到最终向量
        for i in range(min(len(obs_list), max_obs)):
            final_vector[i * feature_dim:(i + 1) * feature_dim] = obs_list[i][1]

        return final_vector

    def control_vel(self, cmd):
        [Vx, Vw, mpc_Vz] = cmd
        Vw_degree = Vw * 180 / np.pi
        
        # --- 修复：信任 MPC 规划的 Z 轴速度，实现真实的 3D 飞行 ---
        # 移除原有的强制 P 控制器定高逻辑 (vz_fixed)。
        # MPC 已经通过 cost_function 考虑了 current_z 到 target_z 的误差以及障碍物。
        # 为了保证底层执行的安全与平滑，我们仅对 mpc_Vz 进行最大速度限幅。
        vz_real = np.clip(mpc_Vz, -0.5, 0.5) 

        # 获取无人机当前姿态
        state = self.client.simGetVehiclePose(vehicle_name=f"Drone_{self.index}")
        pitch, roll, yaw = airsim.to_eularian_angles(state.orientation)
        
        vx_global = Vx * math.cos(yaw)
        vy_global = Vx * math.sin(yaw)
        
        self.client.moveByVelocityAsync(
            vx=vx_global,  
            vy=vy_global,  
            vz=vz_real,    # 使用限幅后的 MPC 真实输出
            duration=1.5,  
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=Vw_degree),  
            vehicle_name=f"Drone_{self.index}"  
        )
        
        # 记录真实的控制输入
        self.current_control = [Vx, Vw, vz_real]

    def generate_goal_point(self, goal):
        """
        初始化目标点，并记录本回合的初始物理基准。
        """
        self.goal = goal
        
        # 获取当前位置到目标的机体坐标向量
        local_goal, _ = self.get_local_goal_and_speed()
        
        # 1. 计算真实的起始 3D 欧氏距离
        dist = np.linalg.norm(local_goal)
        
        # 2. 用于势能奖励 r_potential 的分母 (initial_distance)
        self.initial_distance = dist 
        
        # 用于进度奖励 progress 的减数 (pre_distance)
        # 确保第一步计算进度时，减去的是本回合的起点距离
        self.pre_distance = dist 
        
        # 保留原有的 distance 属性兼容性
        self.distance = dist
        
        return self.distance


    def calculate_dynamic_risk(self):
        """
        计算所有前方障碍物的安全因子和质心距离
        返回:
            risk_tanh_list: Agent1 使用，tanh映射，范围(-1,1)，负值危险正值安全
            risk_relu_list: Agent2 使用，ReLU映射，范围(-∞,0]，0为安全，负值为危险
            distance_list:  每个前方障碍物的质心直线距离，单位米
        """
        risk_tanh_list = []
        risk_relu_list = []
        distance_list = []

        current_pos = np.array(self.get_state(return_index=1))
        state = self.client.getMultirotorState(
            vehicle_name="Drone_" + str(self.index)
        )
        vel = state.kinematics_estimated.linear_velocity
        v_vec_2d = np.array([vel.x_val, vel.y_val])
        v_norm_2d = np.linalg.norm(v_vec_2d)

        # 如果无人机没有用移动，返回空列表
        if v_norm_2d < 0.01:
            return risk_tanh_list, risk_relu_list, distance_list

        v_hat_2d = v_vec_2d / v_norm_2d 
        r_drone = 0.6
        # print(f"计算奖励的障碍物：{self.obstacles}")
        for obs in self.obstacles:
            obs_pos = obs["center"]
            r_obs = obs.get("radius", 0.5)
            r_safe = obs.get("safe_radius", 0.5)

            d_vec_2d = obs_pos[:2] - current_pos[:2]
            d_norm_2d = np.linalg.norm(d_vec_2d)

            if d_norm_2d < 1e-3:
                continue

            if np.dot(d_vec_2d, v_hat_2d) < -0.0:
                continue

            cross_2d = np.cross(d_vec_2d, v_hat_2d)
            # 在打印 d_lat 之前，把向量打出来看看
            # print(f"Velocity Vec: [{v_vec_2d[0]:.2f}, {v_vec_2d[1]:.2f}")
            # print(f"Distance Vec: [{d_vec_2d[0]:.2f}, {d_vec_2d[1]:.2f}")
            d_lat = abs(cross_2d)
            # print(f"d_norm={d_norm_2d:5f}")
            # print(f"d_lat={d_lat:5f}")

            # 净侧向间距
            danger_threshold = r_obs + r_drone + r_safe
            lat_gap = d_lat - danger_threshold
            # print(f"lat_gap={lat_gap:5f}")
            
            # Agent1
            # 安全区（lat_gap >= 0）：tanh 映射，范围 (0, 1)，值越大越安全
            # 危险区（lat_gap < 0） ：线性映射，范围 [-1, 0]，值越大越危险
            if lat_gap >= 0:
                risk_tanh = float(np.tanh(lat_gap))
            else:
                risk_tanh = min(lat_gap / danger_threshold, 0.0)
            # print(f"risk_tanh={risk_tanh:5f}")

                
            # Agent2
            risk_relu = min(lat_gap / danger_threshold, 0.0)
            # print(f"risk_relu={risk_relu:5f}")

            risk_tanh_list.append(risk_tanh)
            risk_relu_list.append(risk_relu)
            distance_list.append(float(d_norm_2d))
            # print("------------------------")

        return risk_tanh_list, risk_relu_list, distance_list

    def get_reward_terminate_result(self, t, update_flag=False, rule_triggered=False, q_weight=0.0, is_crashed=False):

        # 1. 基础物理量
        current_pos = np.array(self.get_state(return_index=1))
        dist_curr = np.linalg.norm(current_pos - np.array(self.goal))

        # 航向角误差（机体坐标系）
        local_goal_vec, _ = self.get_local_goal_and_speed()
        yaw_err = abs(math.atan2(local_goal_vec[1], local_goal_vec[0]))
        # yaw_err ∈ [0, π]，归一化到 [0, 1] 后取负作为惩罚
        yaw_err = -(yaw_err / math.pi)   # 正对目标时为 0，完全反向时为 -1

        # 2. 核心事件判定
        terminate = False
        r_event = 0.0
        result = 'Running'

        if dist_curr < 2.5:
            r_event = 20.0
            terminate = True
            result = 'Reach Goal'
        elif is_crashed:
            r_event = -10.0
            terminate = True
            result = 'Crashed'
        elif t >= self.max_timesteps:
            r_event = -10.0
            terminate = True
            result = 'Time out'

        # 3. 计算安全因子和质心距离
        _, risk_relu_list, _ = self.calculate_dynamic_risk()
        
        # Agent1 奖励：安全-导航权衡形式
        lambda_nav = 3.0
        lambda_safe = 3.0

        if len(risk_relu_list) == 0:
            safety_mpc_for_q = 0.0
        else:
            safety_mpc_for_q = sum(risk_relu_list)

        R_danger_q = max(0.0, -safety_mpc_for_q)
        R_danger_q = min(R_danger_q, 1.0)

        r_weight = -lambda_nav * (q_weight ** 2) - lambda_safe * R_danger_q * ((1.0 - q_weight) ** 2)

        r1 = r_weight + r_event

        # Agent2
        if len(risk_relu_list) == 0:
            safety_mpc = 0.0
        else:
            safety_mpc = sum(risk_relu_list)

        # Agent2 奖励
        if len(risk_relu_list) == 0:
            safety_mpc = 0.0
        else:
            safety_mpc = sum(risk_relu_list)

        # Agent2 奖励参数
        C_trigger = 0.1
        C_miss_rule = 0.5

        w_safe = 1.0
        w_yaw = 1.0

        R_safe = max(0.0, -safety_mpc)
        R_yaw = max(0.0, -yaw_err)

        has_front_obstacle = len(risk_relu_list) > 0
        r_yaw = -w_yaw * R_yaw
        r_safe = -w_safe * R_safe - R_yaw

        if update_flag:
            r_agent2 = -C_trigger
        else:
            if rule_triggered:
                r_agent2 = -C_miss_rule
            else:
                if has_front_obstacle:
                    # 有前方有效障碍物：只根据安全因子计算奖励
                    r_agent2 = r_safe
                else:
                    # 没有前方有效障碍物：只根据偏航角计算奖励
                    r_agent2 = r_yaw

        r2 = r_agent2 + r_event

        # 6. 更新状态缓存
        self.pre_distance = dist_curr
        self.pre_control = list(self.current_control)

        # 7. 记录奖励日志
        if self.log_reward:
            self.reward_logger.info(
                f'{self.current_epoch}, ' # 回合
                f'{t}, ' # 步数
                f'{result}, ' # 结果
                f'{dist_curr:.5f}, ' # 距离
                f'{len(risk_relu_list)}, ' # 障碍物数量
                f'{yaw_err:.5f}, ' # 偏航角
                f'{q_weight:.5f}, ' # 当前权重
                f'{r_weight:.5f}, ' # 权重奖励
                f'{r1:.5f}, ' # r1总奖励
                f'{r_safe:5f}, ' # 有障碍物奖励
                f'{r_yaw:.5f}, ' # 无障碍物奖励
                f'{r2:.5f}, '  # r2总奖励
                f'"{risk_relu_list}"' # 安全因子列表
            )

        return float(r1), float(r2), terminate, result

    def plot_path(self):
        color_rgba = [1.0, 0.0, 0.0, 1.0]  # RGBA
        num_points = len(self.record_pos)
        trajecy_point = []
        for i in range(num_points):
            # if env.index != 0:
            trajecy_point.append(airsim.Vector3r(self.record_pos[i][0], self.record_pos[i][1] + 2 * float(self.index),
                                                 self.record_pos[i][2]))
            # else:
            #     trajecy_point.append(airsim.Vector3r(self.record_pos[i][0], self.record_pos[i][1], self.record_pos[i][2]))
        # 绘制轨迹
        for i in range(num_points - 1):
            next_index = i + 1
            self.client.simPlotLineList([trajecy_point[i], trajecy_point[next_index]],
                                        color_rgba, 15, is_persistent=True)

    def plot_predicted_trajectory(self, predicted_states, color_rgba=[0.0, 1.0, 0.0, 1.0], thickness=5, duration=5.0):
        """
        绘制MPC预测的轨迹

        参数:
        predicted_states: MPC预测的状态序列，形状为 (horizon+1, state_dim)
                          每行包含 [x, y, z, yaw, vx, wz, vz]
        color_rgba: 轨迹颜色 [r, g, b, a]
        thickness: 线条粗细
        duration: 显示持续时间（秒），如果为None则永久显示
        """
        if predicted_states is None or len(predicted_states) < 2:
            return

        # 创建轨迹点列表
        trajectory_points = []
        for i in range(len(predicted_states)):
            state = predicted_states[i]
            # 为不同环境实例添加y方向偏移，避免轨迹重叠
            y_offset = 2 * float(self.index) if self.index != 0 else 0
            point = airsim.Vector3r(
                float(state[0]),  # x
                float(state[1]) + y_offset,  # y + offset
                float(state[2])   # z
            )
            trajectory_points.append(point)

        # 绘制轨迹线段
        for i in range(len(trajectory_points) - 1):
            line_points = [trajectory_points[i], trajectory_points[i + 1]]
            self.client.simPlotLineList(line_points, color_rgba, thickness, is_persistent=False,duration=duration)

    def get_obstacles(self):
        """
        获取静态障碍物信息并转换为MPC控制器可用的格式。
        该函数通常在 reset_world 之后调用一次，作为环境的静态真值地图。
        """
        self.all_obstacles = [] # 清空重新获取
        obstacles = []

        # --- 1. 地图配置 (Map Configuration) ---
        map_configs = {
            101: {
                "static":  {"num": 100, "r": 0.5, "safe": 0.5}, 
                "dynamic": {"num": 0, "r": 0.0, "safe": 1.2}
            },
            102: {
                "static":  {"num": 100, "r": 1.0, "safe": 0.5}, 
                "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}
            },
            103: {
                "static":  {"num": 100, "r": 0.5, "safe": 0.5}, 
                "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}
            },
            104: {
                "static":  {"num": 100, "r": 0.5, "safe": 0.5}, 
                "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}
            }
        }

        # 获取当前地图配置
        cfg = map_configs.get(self.map_index)
        if not cfg:
            print(f"[Warning] Map index {self.map_index} not defined in get_obstacles.")
            return

        # --- 2. 静态障碍物获取 ---
        num_static = cfg["static"]["num"]
        if num_static > 0:
            for i in range(1, num_static + 1):
                try:
                    obj_name = f"Obstacle_{i}"
                    state = self.client.simGetObjectPose(object_name=obj_name)
                    pos = state.position
                    
                    if not np.isnan(pos.x_val):
                        obstacles.append({
                            "center": np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)]),
                            "radius": cfg["static"]["r"],
                            "safe_radius": cfg["static"]["safe"],
                            "weight": 1.0,
                            "type": "static",
                            "name": obj_name
                        })
                except Exception:
                    continue

        # --- 3. 动态障碍物获取 ---
        num_dynamic = cfg["dynamic"]["num"]
        if num_dynamic > 0:
            for i in range(1, num_dynamic + 1):
                try:
                    obj_name = f"D_Obstacle_{i}"
                    if obj_name in self.dynamic_origin_cache:
                        origin_pos = self.dynamic_origin_cache[obj_name].copy()
                    else:
                        state = self.client.simGetObjectPose(object_name=obj_name)
                        pos = state.position
                        origin_pos = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
                        self.dynamic_origin_cache[obj_name] = origin_pos
                    
                    obs_data = {
                        "center": origin_pos.copy(),
                        "radius": cfg["dynamic"]["r"],
                        "safe_radius": cfg["dynamic"]["safe"],
                        "weight": 1.5,
                        "type": "dynamic",
                        "name": obj_name
                    }
                    obstacles.append(obs_data)
                except Exception as e:
                    print(f"Error loading {obj_name}: {e}")
                    continue

        # --- 4. 最终保存 ---
        self.all_obstacles = obstacles

    def move_obstacles(self, step):
        """
        移动场景中的动态障碍物（匀速线性模型）。
        每步沿 X 轴移动固定距离，位置从内存追踪，避免从仿真器累积读取误差。
        同步更新 all_obstacles 中的 center，供碰撞检测和奖励计算使用。
        """
        map_configs = {
            11: {"dynamic": {"num": 3}},
            12: {"dynamic": {"num": 3}},
            101: {"dynamic": {"num": 0}},
            102: {"dynamic": {"num": 0}},
            103: {"dynamic": {"num": 0}},
            104: {"dynamic": {"num": 0}},
            105: {"dynamic": {"num": 10}}
        }

        cfg = map_configs.get(self.map_index)
        if not cfg or cfg["dynamic"]["num"] == 0:
            return  # 当前地图没有动态障碍物，直接返回

        num_dynamic = cfg["dynamic"]["num"]
        move_dist = -0.4  # 每步沿 X 轴移动量（米）

        for i in range(1, num_dynamic + 1):
            obj_name = f"D_Obstacle_{i}"
            try:
                # 1. 首次出现：从仿真器读取初始位置并缓存
                if obj_name not in self._dynamic_positions:
                    state = self.client.simGetObjectPose(object_name=obj_name)
                    pos = state.position
                    init_pos = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
                    self._dynamic_positions[obj_name] = init_pos.copy()
                    if obj_name not in self.dynamic_origin_cache:
                        self.dynamic_origin_cache[obj_name] = init_pos.copy()

                # 2. 从内存读取当前位置，计算新位置
                new_pos = self._dynamic_positions[obj_name].copy()
                new_pos[0] += move_dist

                # 3. 更新内存
                self._dynamic_positions[obj_name] = new_pos

                # 4. 同步更新 all_obstacles 中该障碍物的 center（供碰撞检测、奖励计算使用）
                for obs in self.all_obstacles:
                    if obs.get("name") == obj_name:
                        obs["center"] = new_pos.copy()
                        break

                # 5. 将新位置写回仿真器（teleport=True 确保对象真正移动）
                orig_state = self.client.simGetObjectPose(object_name=obj_name)
                new_pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        float(new_pos[0]),
                        float(new_pos[1]),
                        float(new_pos[2])
                    ),
                    orientation_val=orig_state.orientation
                )
                self.client.simSetObjectPose(object_name=obj_name, pose=new_pose, teleport=True)

            except Exception as e:
                # print(f"[Move] Failed to move {obj_name}: {e}")
                continue


import math
import copy
import torch
import numpy as np
import pandas as pd
import random
import time

class RunningMeanStd:
    def __init__(self, min_val, max_val):
        # 归一化方法，这里采取（输入-最小值）/（最大值-最小值）
        self.min = np.array(min_val, dtype='float32')
        self.max = np.array(max_val, dtype='float32')
        self.count = 1e-4

    def normalize(self, x):
        """
        归一化采用最大值最小值方法：(当前值 - 最小值) / (最大值 - 最小值)。
        """
        norm_x = (x - self.min) / (self.max - self.min + 1e-8)
        return norm_x

class ReplayBuffer_qmpc:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['o_z', 'o_q', 'o_velocity', 'action','next_o_z', 'next_o_q', 'next_o_velocity','reward', 'not_done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        batch = []
        for field in self.memory.columns:
            samples = self.memory.loc[indices, field].values
            try:
                stacked = np.stack(samples)
                batch.append(stacked)
            except ValueError as e:
                # 打印出错的字段和具体的形状，方便排查
                print(f"Error stacking field: {field}")
                for i, s in enumerate(samples):
                    print(f"Sample {i} shape: {np.shape(s)}")
                raise e
        return batch

    def save_to_file(self, filename):
        """保存replay buffer到CSV文件"""
        # 只保存已有的数据，不包括空的行
        data_to_save = self.memory.iloc[:self.count].copy()
        
        # 处理数组数据，确保能正确保存到CSV
        for col in data_to_save.columns:
            if data_to_save[col].dtype == 'object':
                # 将numpy数组转换为字符串格式，保持原始形状
                data_to_save[col] = data_to_save[col].apply(
                    lambda x: str(x.tolist()) if hasattr(x, 'tolist') else str(x)
                )
        
        data_to_save.to_csv(filename, index=False)
        print(f"Replay buffer saved to {filename} with {self.count} samples")

    def load_from_file(self, filename):
        """从CSV文件加载replay buffer"""
        try:
            memory_df = pd.read_csv(filename)
            
            expected_columns = ['o_z', 'o_q', 'o_velocity', 'action', 'next_o_z', 'next_o_q', 'next_o_velocity', 'reward', 'not_done']
            if list(memory_df.columns) != expected_columns:
                print(f"Warning: Column names don't match. Expected: {expected_columns}")
                print(f"Found: {list(memory_df.columns)}")
            
            self.memory = pd.DataFrame(index=range(self.capacity), columns=expected_columns)
            
            import ast
            for i, row in memory_df.iterrows():
                for col in expected_columns:
                    if col not in memory_df.columns:
                        continue
                    value = row[col]
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            value = np.array(ast.literal_eval(value), dtype=np.float32)
                        except Exception as e:
                            print(f"Warning: Failed to convert {col} at row {i}: {e}")
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    self.memory.loc[i, col] = value
            
            self.count = len(memory_df)
            self.i = self.count % self.capacity
            
            print(f"Replay buffer loaded from {filename} with {self.count} samples")
            return True
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            return False

    def get_size(self):
        """获取当前存储的数据量"""
        return self.count

    def is_full(self):
        """检查buffer是否已满"""
        return self.count == self.capacity

class ReplayBuffer_eMpc:
    """ReplayBuffer for SAC_Ae_eMpc - stores image, yaw, velocity, action, reward, not_done"""
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['o_z', 'o_yaw', 'o_velocity', 'action', 'next_o_z', 'next_o_yaw', 'next_o_velocity', 'reward', 'not_done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        batch = []
        for field in self.memory.columns:
            samples = self.memory.loc[indices, field].values
            try:
                stacked = np.stack(samples)
                batch.append(stacked)
            except ValueError as e:
                # 打印出错的字段和具体的形状，方便排查
                print(f"Error stacking field: {field}")
                for i, s in enumerate(samples):
                    print(f"Sample {i} shape: {np.shape(s)}")
                raise e
        return batch

    def save_to_file(self, filename):
        """保存replay buffer到CSV文件"""
        # 只保存已有的数据，不包括空的行
        data_to_save = self.memory.iloc[:self.count].copy()
        
        # 处理数组数据，确保能正确保存到CSV
        for col in data_to_save.columns:
            if data_to_save[col].dtype == 'object':
                # 将numpy数组转换为字符串格式，保持原始形状
                data_to_save[col] = data_to_save[col].apply(
                    lambda x: str(x.tolist()) if hasattr(x, 'tolist') else str(x)
                )
        
        data_to_save.to_csv(filename, index=False)
        print(f"Replay buffer saved to {filename} with {self.count} samples")

    def load_from_file(self, filename):
        """从CSV文件加载replay buffer"""
        try:
            memory_df = pd.read_csv(filename)
            
            expected_columns = ['o_z', 'o_yaw', 'o_velocity', 'action', 'next_o_z', 'next_o_yaw', 'next_o_velocity', 'reward', 'not_done']
            if list(memory_df.columns) != expected_columns:
                print(f"Warning: Column names don't match. Expected: {expected_columns}")
                print(f"Found: {list(memory_df.columns)}")
            
            self.memory = pd.DataFrame(index=range(self.capacity), columns=expected_columns)
            
            import ast
            for i, row in memory_df.iterrows():
                for col in expected_columns:
                    if col not in memory_df.columns:
                        continue
                    value = row[col]
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            value = np.array(ast.literal_eval(value), dtype=np.float32)
                        except Exception as e:
                            print(f"Warning: Failed to convert {col} at row {i}: {e}")
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    self.memory.loc[i, col] = value
            
            self.count = len(memory_df)
            self.i = self.count % self.capacity
            
            print(f"Replay buffer loaded from {filename} with {self.count} samples")
            return True
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            return False

    def get_size(self):
        """获取当前存储的数据量"""
        return self.count

    def is_full(self):
        """检查buffer是否已满"""
        return self.count == self.capacity


class Statistics:
    def __init__(self, capacity=100):
        """
        MPC导航统计类
        """
        self.capacity = capacity
        
        # 基本统计
        self.success_count = 0
        self.fail_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        
        # MPC相关统计
        self.mpc_update_count = 0
        self.mpc_failure_count = 0
        self.mpc_reuse_count = 0
        self.total_mpc_computations = 0
        self.total_control_commands = 0
        
        # 轨迹统计
        self.total_steps = 0
        self.total_distance = 0.0
        self.total_straight_distance = 0.0
        self.total_extra_distance_ratio = 0.0
        self.total_jerk_rms = 0.0
        
        # [修改] 偏航角误差与平均最小障碍物距离统计
        self.total_avg_yaw_error = 0.0           # 累积每个回合的平均偏航角误差
        self.total_avg_min_obs_distance = 0.0    # 累积每个回合的平均最小障碍物距离
        
        # 滑动窗口统计 (最近capacity次的记录)
        from collections import deque
        self.recent_results = deque(maxlen=capacity)
        self.recent_steps = deque(maxlen=capacity)
        self.recent_distances = deque(maxlen=capacity)
        self.recent_mpc_updates = deque(maxlen=capacity)
        self.recent_straight_distances = deque(maxlen=capacity)
        self.recent_extra_distances = deque(maxlen=capacity)
        self.recent_jerk_rms = deque(maxlen=capacity)
        
        # [修改] 滑动窗口记录
        self.recent_avg_yaw_errors = deque(maxlen=capacity)
        self.recent_avg_min_obs_distances = deque(maxlen=capacity)
        
        self.episode_count = 0

        # 逐步序列数据（每个元素对应一个episode）
        self.episodes_trajectory = []       # 每步位置 [(x, y, z), ...]
        self.episodes_velocity = []         # 每步速度 [(vx, vy, vz), ...]
        self.episodes_acceleration = []     # 每步加速度 [(ax, ay, az), ...]
        self.episodes_trigger_seq = []
        self.episodes_dist_to_goal = []
        self.episodes_q_weights = []
        self.episodes_avg_yaw_error = []       
        self.episodes_avg_min_obs_distance = []
        
    def record_episode(self, success, steps, distance=0.0,
                       mpc_updates=0, mpc_failures=0, mpc_reuses=0,
                       mpc_computations=0, result_type='unknown',
                       straight_distance=None, jerk_rms=None,
                       avg_yaw_error=None, avg_min_obs_distance=None,  
                       trajectory=None, velocities=None, accelerations=None,
                       trigger_seq=None,
                       dist_to_goal=None, q_weights=None):
        """记录一个episode的统计数据"""
        self.episode_count += 1
        
        if success: self.success_count += 1
        else: self.fail_count += 1
            
        if result_type == 'collision': self.collision_count += 1
        elif result_type == 'timeout': self.timeout_count += 1
            
        self.mpc_update_count += mpc_updates
        self.mpc_failure_count += mpc_failures
        self.mpc_reuse_count += mpc_reuses
        self.total_mpc_computations += mpc_computations
        self.total_control_commands += steps
        self.total_steps += steps
        self.total_distance += distance
        
        if straight_distance is not None and straight_distance > 0.0:
            self.total_straight_distance += straight_distance
            ed = distance / straight_distance if distance > 0.0 else 0.0
            self.total_extra_distance_ratio += ed
            
        if jerk_rms is not None and jerk_rms >= 0.0:
            self.total_jerk_rms += jerk_rms

        # [修改] 更新偏航角误差和平均最小障碍物距离
        if avg_yaw_error is not None and avg_yaw_error >= 0.0:
            self.total_avg_yaw_error += avg_yaw_error
        if avg_min_obs_distance is not None and avg_min_obs_distance >= 0.0:
            self.total_avg_min_obs_distance += avg_min_obs_distance
        
        self.recent_results.append(1 if success else 0)
        self.recent_steps.append(steps)
        self.recent_distances.append(distance)
        self.recent_mpc_updates.append(mpc_updates)
        
        if straight_distance is not None and straight_distance > 0.0:
            self.recent_straight_distances.append(straight_distance)
            ed = distance / straight_distance if distance > 0.0 else 0.0
            self.recent_extra_distances.append(ed)
            
        if jerk_rms is not None and jerk_rms >= 0.0:
            self.recent_jerk_rms.append(jerk_rms)
            
        # [修改] 存入滑动窗口
        if avg_yaw_error is not None and avg_yaw_error >= 0.0:
            self.recent_avg_yaw_errors.append(avg_yaw_error)
        if avg_min_obs_distance is not None and avg_min_obs_distance >= 0.0:
            self.recent_avg_min_obs_distances.append(avg_min_obs_distance)

        # 存储逐步序列数据
        if success:
            self.episodes_trajectory.append(trajectory if trajectory is not None else [])
            self.episodes_velocity.append(velocities if velocities is not None else [])
            self.episodes_acceleration.append(accelerations if accelerations is not None else [])
            self.episodes_trigger_seq.append(trigger_seq if trigger_seq is not None else [])
            self.episodes_dist_to_goal.append(dist_to_goal if dist_to_goal is not None else [])
            self.episodes_q_weights.append(q_weights if q_weights is not None else [])
            self.episodes_avg_yaw_error.append(avg_yaw_error if avg_yaw_error is not None else 0.0)
            self.episodes_avg_min_obs_distance.append(avg_min_obs_distance if avg_min_obs_distance is not None else 0.0)

    # ---------------- 基础获取方法省略(保持不变) ----------------
    def get_success_rate(self, recent=False):
        if recent and len(self.recent_results) > 0: return sum(self.recent_results) / len(self.recent_results)
        else: return self.success_count / (self.success_count + self.fail_count) if (self.success_count + self.fail_count) > 0 else 0.0
            
    def get_average_steps(self, recent=False):
        if recent and len(self.recent_steps) > 0: return sum(self.recent_steps) / len(self.recent_steps)
        else: return self.total_steps / self.episode_count if self.episode_count > 0 else 0.0
            
    def get_average_distance(self, recent=False):
        if recent and len(self.recent_distances) > 0: return sum(self.recent_distances) / len(self.recent_distances)
        else: return self.total_distance / self.success_count if self.success_count > 0 else 0.0

    def get_average_extra_distance(self, recent=False):
        if recent and len(self.recent_extra_distances) > 0: return sum(self.recent_extra_distances) / len(self.recent_extra_distances)
        else: return (self.total_extra_distance_ratio / self.success_count) if self.success_count > 0 else 0.0

    def get_average_jerk_rms(self, recent=False):
        if recent and len(self.recent_jerk_rms) > 0: return sum(self.recent_jerk_rms) / len(self.recent_jerk_rms)
        else: return (self.total_jerk_rms / self.success_count) if self.success_count > 0 else 0.0
            
    def get_average_mpc_updates(self, recent=False):
        if recent and len(self.recent_mpc_updates) > 0: return sum(self.recent_mpc_updates) / len(self.recent_mpc_updates)
        else: return self.mpc_update_count / self.episode_count if self.episode_count > 0 else 0.0
            
    def get_mpc_success_rate(self):
        total_mpc = self.mpc_update_count + self.mpc_failure_count
        return (self.mpc_update_count / total_mpc) if total_mpc > 0 else 0.0
        
    def get_computation_efficiency(self):
        return (self.total_mpc_computations / self.total_steps) if self.total_steps > 0 else 0.0

    # [修改] 获取平均偏航角误差
    def get_average_yaw_error(self, recent=False):
        if recent and len(self.recent_avg_yaw_errors) > 0:
            return sum(self.recent_avg_yaw_errors) / len(self.recent_avg_yaw_errors)
        else:
            return (self.total_avg_yaw_error / self.success_count) if self.success_count > 0 else 0.0

    # [修改] 获取整个测试期间的平均最小障碍物距离
    def get_average_avg_min_obs_distance(self, recent=False):
        if recent and len(self.recent_avg_min_obs_distances) > 0:
            return sum(self.recent_avg_min_obs_distances) / len(self.recent_avg_min_obs_distances)
        else:
            return (self.total_avg_min_obs_distance / self.success_count) if self.success_count > 0 else 0.0

    def print_statistics(self, recent=False, verbose=False):
        print("\n" + "="*60)
        if recent: print(f"📊 最近 {len(self.recent_results)} 次任务统计")
        else: print(f"📊 总体统计 (共 {self.episode_count} 次任务)")
        print("="*60)
        
        success_rate = self.get_success_rate(recent)
        print(f"✅ 成功率: {success_rate*100:.2f}% ", end="")
        if not recent: print(f"({self.success_count}/{self.success_count + self.fail_count})")
        else: print()
        
        avg_ed = self.get_average_extra_distance(recent)
        if avg_ed > 0.0: print(f"📏 平均额外距离 ED: {avg_ed:.3f} (L_actual / L_straight)")
        else: print("📏 平均额外距离 ED: N/A")

        avg_jerk = self.get_average_jerk_rms(recent)
        if avg_jerk > 0.0: print(f"🎯 平均 RMS Jerk: {avg_jerk:.4f}")
        else: print("🎯 平均 RMS Jerk: N/A")

        ce = self.get_computation_efficiency()
        print(f"⚡ 计算效率 CE: {ce*100:.2f}%")
        
        if verbose and not recent:
            print("\n" + "-"*60)
            print("详细统计:")
            print(f"  碰撞次数: {self.collision_count}")
            print(f"  超时次数: {self.timeout_count}")
            print(f"  MPC更新/失败/重用: {self.mpc_update_count} / {self.mpc_failure_count} / {self.mpc_reuse_count}")
            print(f"  总飞行距离: {self.total_distance:.2f}m")
            print(f"  总直线距离: {self.total_straight_distance:.2f}m")
            
        print("="*60 + "\n")
        
    def get_statistics_dict(self, recent=False):
        stats = {
            'episode_count': self.episode_count if not recent else len(self.recent_results),
            'success_rate': self.get_success_rate(recent),
            'success_count': self.success_count,
            'fail_count': self.fail_count,
            'collision_count': self.collision_count,
            'timeout_count': self.timeout_count,
            'avg_steps': self.get_average_steps(recent),
            'avg_distance': self.get_average_distance(recent),
            'avg_mpc_updates': self.get_average_mpc_updates(recent),
            'avg_extra_distance': self.get_average_extra_distance(recent),
            'avg_jerk_rms': self.get_average_jerk_rms(recent),
            'computation_efficiency': self.get_computation_efficiency(),
        }
        
        if not recent:
            stats.update({
                'total_steps': self.total_steps,
                'total_distance': self.total_distance,
                'mpc_update_count': self.mpc_update_count,
                'mpc_failure_count': self.mpc_failure_count,
                'mpc_reuse_count': self.mpc_reuse_count,
                'mpc_success_rate': self.get_mpc_success_rate(),
                'total_mpc_computations': self.total_mpc_computations,
                'total_control_commands': self.total_control_commands,
                'episodes_trajectory': self.episodes_trajectory,
                'episodes_velocity': self.episodes_velocity,
               'episodes_acceleration': self.episodes_acceleration, 
                'episodes_trigger_seq': self.episodes_trigger_seq,
                'episodes_dist_to_goal': self.episodes_dist_to_goal,
                'episodes_q_weights': self.episodes_q_weights,
            })
            
        return stats
        
    def reset(self):
        self.__init__(self.capacity)


#----------------------------------------------------
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def global2body(roll, pitch, yaw, g_goal, g_drone):
    #构建旋转矩阵，用于全局坐标系转换到无人机的局部坐标系
    R_x = np.array([[1, 0, 0],[0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_g2b = np.matmul(R_z, R_y)
    R_g2b = np.matmul(R_g2b, R_x)
    return np.matmul(R_g2b, (g_goal - g_drone))

def generate_points(num_env=20, map_index=1, fixed=False):
    """
    生成训练/测试用的起始点和目标点。
    :param num_env: 环境数量
    :param map_index: 地图索引
    :param fixed: 是否固定起点和终点（用于对比轨迹）
    """
    # 默认 Z 轴范围
    default_z_range = (-3.0, -2.0)
    default_z_range1 = (-1.5, -1.0)
    
    # 根据地图索引配置范围
    # 静态地图测试
    if map_index == 102:
        x_init_range, y_init_range, z_init_range = (9.0, 10.0), (17.0, 18.0), default_z_range
        x_goal_range, y_goal_range, z_goal_range = (60.0, 71.0), (7.0, 18.0), default_z_range
        distance_threshold_min, distance_threshold_max = 51.0, 52.0
    # 动态地图测试（还需要修改）
    elif map_index == 103:
        x_init_range, y_init_range, z_init_range = (9.0, 10.0), (17.0, 18.0), default_z_range
        x_goal_range, y_goal_range, z_goal_range = (60.0, 61.0), (17.0, 18.0), default_z_range
        distance_threshold_min, distance_threshold_max = 51.0, 53.0
    # 绘图地图测试
    elif map_index == 104:
        x_init_range, y_init_range, z_init_range = (9.0, 10.0), (17.0, 18.0), default_z_range
        x_goal_range, y_goal_range, z_goal_range = (60.0, 61.0), (17.0, 18.0), default_z_range
        distance_threshold_min, distance_threshold_max = 51.0, 53.0
    # 训练地图
    else:
        x_init_range, y_init_range, z_init_range = (9.0, 10.0), (0.0, 28.0), default_z_range
        x_goal_range, y_goal_range, z_goal_range = (60.0, 61.0), (0.0, 28.0), default_z_range
        distance_threshold_min, distance_threshold_max = 51.0, 53.0

    init_points = []
    goal_points = []

    if fixed:
        # 取范围的中点作为固定坐标
        fixed_init = [np.mean(x_init_range), np.mean(y_init_range), np.mean(z_init_range)]
        fixed_goal = [np.mean(x_goal_range), np.mean(y_goal_range), np.mean(z_goal_range)]
        
        init_points = [fixed_init for _ in range(num_env)]
        goal_points = [fixed_goal for _ in range(num_env)]
        return init_points, goal_points

    # 原有的随机生成逻辑
    for _ in range(num_env):
        while True:
            rx_init = random.uniform(*x_init_range)
            ry_init = random.uniform(*y_init_range)
            rz_init = random.uniform(*z_init_range)
            
            rx_goal = random.uniform(*x_goal_range)
            ry_goal = random.uniform(*y_goal_range)
            rz_goal = random.uniform(*z_goal_range)

            init_point = (rx_init, ry_init, rz_init)
            goal_point = (rx_goal, ry_goal, rz_goal)

            distance = math.sqrt((init_point[0] - goal_point[0]) ** 2 + 
                                 (init_point[1] - goal_point[1]) ** 2 + 
                                 (init_point[2] - goal_point[2]) ** 2)
            
            if distance_threshold_min <= distance <= distance_threshold_max:
                break
        
        init_points.append(list(init_point))
        goal_points.append(list(goal_point))

    return init_points, goal_points


    
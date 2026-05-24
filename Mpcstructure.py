from cgi import print_form
import math
from tkinter import N, NO
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class MpcStructure:
    def __init__(self, dt=0.1, horizon=10, Q=None, max_velocity=5.0, max_acceleration=2.0, max_yaw_acceleration=0.5, max_yaw_rate=1.0,
                 position_scale=10.0, accel_scale=None, yaw_accel_scale=None,
                 roll_scale=np.deg2rad(10.0), pitch_scale=np.deg2rad(10.0), yaw_scale=np.deg2rad(10.0),
                 q_roll=0.0, q_pitch=0.2, q_yaw=0.2,
                 lambda_omega=0.0, yaw_rate_scale=None):
        """
        四旋翼无人机MPC控制器
        
        参数:
        dt: 采样时间
        horizon: 预测时域长度
        max_velocity: 最大速度限制
        max_acceleration: 最大加速度限制（线加速度 ax, az）
        max_yaw_acceleration: 最大角加速度限制（aw）
        max_yaw_rate: 最大角速度限制（|wz|）
        position_scale: 位置误差归一化尺度（米）
        accel_scale: 线加速度归一化尺度（默认= max_acceleration）
        yaw_accel_scale: 角加速度归一化尺度（默认= max_yaw_acceleration）
        """
        self.dt = dt
        self.horizon = horizon
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_yaw_acceleration = max_yaw_acceleration
        self.max_yaw_rate = max_yaw_rate
        # 归一化尺度 - 用于MPC成本函数中归一化不同物理量纲的变量
        self.position_scale = float(position_scale)  # 位置误差归一化尺度（米），用于位置跟踪代价
        self.accel_scale = float(accel_scale) if accel_scale is not None else float(max_acceleration)  # 线加速度归一化尺度（ax, az），用于控制平滑代价
        self.yaw_accel_scale = float(yaw_accel_scale) if yaw_accel_scale is not None else float(max_yaw_acceleration)  # 角加速度归一化尺度（aw），用于偏航控制平滑代价
        
        # 未使用的归一化变量 - 保留用于扩展功能
        self.roll_scale = float(roll_scale)    # 横滚角归一化尺度（弧度），当前未使用
        self.pitch_scale = float(pitch_scale)  # 俯仰角归一化尺度（弧度），当前未使用  
        self.yaw_scale = float(yaw_scale)      # 偏航角归一化尺度（弧度），当前未使用
        
        self.q_roll = float(q_roll)
        self.q_pitch = float(q_pitch)
        self.q_yaw = float(q_yaw)
        self.lambda_omega = float(lambda_omega)
        self.yaw_rate_scale = float(yaw_rate_scale) if yaw_rate_scale is not None else float(max_yaw_rate)
        
        # 状态维度: [x, y, z, yaw, vx, wz, vz] (7维)
        self.state_dim = 7
        # 控制输入维度: [ax, aw, az] (前向加速度、偏航角加速度、垂向加速度)
        self.control_dim = 3
        self.control_sequence = None
        
        # 权重矩阵
        self.Q = None # 状态权重，确保是numpy数组
        self.R = None # 控制权重，确保是numpy数组
        self.Qp = None # 位置权重，确保是numpy数组
        self.Qf = None # 终端状态权重，确保是numpy数组
        
    def quaternion_normalize(self, q):
        """四元数归一化"""
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.array([1, 0, 0, 0])
        return q / norm
    
    def _euler_from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """将四元数转换为欧拉角 (roll, pitch, yaw)，采用 ZYX 顺序。"""
        w, x, y, z = q
        # roll (x 轴)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y 轴)
        sinp = 2.0 * (w * y - z * x)
        sinp_clamped = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp_clamped)

        # yaw (z 轴)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """从欧拉角 (roll, pitch, yaw) 生成四元数，采用 ZYX 顺序。"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return self.quaternion_normalize(np.array([w, x, y, z]))

    def _wrap_to_pi(self, angle: float) -> float:
        """将角度包裹到 (-pi, pi] 区间。"""
        wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
        # 将 -pi 统一映射为 pi，避免不连续的双表示
        return np.pi if np.isclose(wrapped, -np.pi) else wrapped

    def _angle_error_smooth(self, angle_error: float) -> float:
        """平滑角度误差：atan2(sin(Δ), cos(Δ))，在±pi附近连续可导。"""
        return np.arctan2(np.sin(angle_error), np.cos(angle_error))

    def state_dynamics(self, state, control):
        """
        无人机简化状态动力学（方案A：平面航迹+航向+高度）
        
        状态: [x, y, z, yaw, vx, wz, vz]
        控制: [ax, aw, az] （aw 为偏航角加速度）
        """
        x, y, z, yaw, vx, wz, vz = state
        ax, aw, az = control

        # 速度与角速度更新
        vx_new = vx + ax * self.dt
        vz_new = vz + az * self.dt
        wz_new = wz + aw * self.dt
        
        # 姿态更新：使用角加速度积分（θ = θ₀ + ω₀·t + ½·α·t²）
        yaw_new = yaw + wz * self.dt + 0.5 * aw * self.dt**2
        
        # 位置更新：使用加速度积分，考虑yaw在dt内的变化（使用平均角度）
        yaw_avg = 0.5 * (yaw + yaw_new)  # dt内的平均yaw角度
        x_new = x + (np.cos(yaw_avg) * vx) * self.dt + (np.cos(yaw_avg) * 0.5 * ax * self.dt**2)
        y_new = y + (np.sin(yaw_avg) * vx) * self.dt + (np.sin(yaw_avg) * 0.5 * ax * self.dt**2)
        z_new = z + vz * self.dt + 0.5 * az * self.dt**2

        return np.array([
            x_new, y_new, z_new,
            yaw_new,
            vx_new, wz_new, vz_new
        ])
    
    def predict_trajectory(self, initial_state, control_sequence):
        """预测轨迹"""
        states = np.zeros((self.horizon + 1, self.state_dim))
        states[0] = initial_state
        
        for i in range(self.horizon):
            states[i + 1] = self.state_dynamics(states[i], control_sequence[i])
        
        return states
    
    def cost_function(self, control_sequence, initial_state, goal_position, obstacles=None):
        """
        MPC成本函数（含归一化）：
        cost1 - 位置到目标
        cost3 - 控制平滑（大小 + 变化率）
        cost4 - 障碍物避障（椭圆势场，动态障碍物按正弦模型预测位置）
        cost5 - 航向角误差（> 120° 时激活）
        
        Args:
            current_step: 当前回合步数（env.step），用于动态障碍物正弦位置预测
        """
        control_sequence = control_sequence.reshape(self.horizon, self.control_dim)
        predicted_states = self.predict_trajectory(initial_state, control_sequence)

        cost1 = 0.0
        cost3 = 0.0
        cost4 = 0.0
        cost5 = 0.0

        current_pos = initial_state[:3]
        dist_curr_to_goal = np.linalg.norm(current_pos - goal_position)
        s_dyn = np.sqrt(max(1.0, dist_curr_to_goal))

        # 1) 位置到目标代价
        for i in range(self.horizon):
            p = predicted_states[i + 1][:3]
            pos_error_norm = (p - goal_position) / s_dyn
            pos_error_norm = pos_error_norm.reshape(-1, 1)
            cost1 += float(pos_error_norm.T @ self.Qp @ pos_error_norm)

        # 2) 控制平滑代价（大小 + 变化率）
        for i in range(self.horizon):
            ax_norm = control_sequence[i, 0] / self.accel_scale
            aw_norm = control_sequence[i, 1] / self.yaw_accel_scale
            az_norm = control_sequence[i, 2] / self.accel_scale
            u_norm  = np.array([ax_norm, aw_norm, az_norm])
            cost3  += float(u_norm.T @ self.R @ u_norm)

            if i < self.horizon - 1:
                delta_ax    = (control_sequence[i+1, 0] - control_sequence[i, 0]) / self.accel_scale
                delta_aw    = (control_sequence[i+1, 1] - control_sequence[i, 1]) / self.yaw_accel_scale
                delta_az    = (control_sequence[i+1, 2] - control_sequence[i, 2]) / self.accel_scale
                delta_u_norm = np.array([delta_ax, delta_aw, delta_az])
                cost3       += float(delta_u_norm.T @ self.R @ delta_u_norm)

        # 3) 障碍物避障代价（椭圆势场）
        #    静态障碍物：位置固定，直接使用 obs["center"]
        #    动态障碍物：按匀速线性模型预测第 i 步时的位置
        #      c[0] = center[0] + (-0.1) * i
        if obstacles:
            for i in range(self.horizon + 1):
                p   = predicted_states[i][:3]
                yaw = predicted_states[i][3]

                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)

                for obs in obstacles:
                    c = obs.get("center")
                    if c is None:
                        continue
                    r_safe   = float(obs.get("safe_radius", 1.0))
                    r_obs    = float(obs.get("radius", 0.5))
                    name     = obs.get("name", "obstacle")
                    obs_type = obs.get("type", "static")

                    # --- 动态障碍物：正弦运动预测 ---
                    if obs_type == "dynamic":
                        c = c.copy()
                        c[0] += (-0.1) * i 

                    # --- 地板代价 ---
                    if name == "floor":
                        dist_z = abs(p[2] - c[2]) - r_safe
                        d_eff  = max(dist_z, 0.0)
                        cost4 += float(self.Q[6]) * 5.0 / (d_eff + 0.2)

                    # --- 通用障碍物（静态 + 动态）椭圆势场 ---
                    else:
                        dx = p[0] - c[0]
                        dy = p[1] - c[1]
                        dz = p[2] - c[2]
                        dx_body =  dx * cos_y + dy * sin_y
                        dy_body = -dx * sin_y + dy * cos_y

                        d_soft  = r_obs + r_safe
                        s_lat   = d_soft ** 2
                        s_long  = s_lat * 4.0
                        epsilon = 0.001
                        w_real  = float(self.Q[6])
                        denom   = (dx_body**2 / s_long) + (dy_body**2 / s_lat) + epsilon
                        cost4  += w_real / denom

        # 4) 航向角误差代价（误差 > 120° 时激活）
        YAW_THRESHOLD = math.radians(120.0)
        for i in range(self.horizon + 1):
            p   = predicted_states[i][:3]
            yaw = predicted_states[i][3]

            pos_err    = goal_position[:2] - p[:2]
            target_yaw = math.atan2(pos_err[1], pos_err[0])
            yaw_err    = abs(math.atan2(
                math.sin(target_yaw - yaw),
                math.cos(target_yaw - yaw)
            ))

            if yaw_err > YAW_THRESHOLD:
                excess  = yaw_err - YAW_THRESHOLD
                cost5  += 5.0 * (excess ** 2)

        return float(cost1) + float(cost3) + float(cost4) + float(cost5)
    
    def solve_mpc(self, current_state, target_position, obstacles=None):
        """
        求解MPC优化问题
        
        参数:
        current_state: 当前状态 [x, y, z, yaw, vx, wz, vz]
        target_position: 目标位置 [x, y, z]
        obstacles: 障碍物列表（可选）
        
        返回:
        optimal_control: 最优控制序列
        predicted_states: 预测状态序列
        """

        dist_to_goal = np.linalg.norm(current_state[:3] - target_position)
        dynamic_scale = np.sqrt(max(1.0, dist_to_goal))

        # 初始控制序列
        # 初始控制序列获取逻辑
        if self.control_sequence is not None:
            # 方案 A：热启动 (Warm Start) - 提取上一次的最优解进行时间轴平移
            shifted_control = np.zeros_like(self.control_sequence)
            shifted_control[:-1] = self.control_sequence[1:] # 把未来的动作提前一步
            shifted_control[-1] = np.zeros(self.control_dim) # 最后一步补 0 (惯性滑行)
            initial_control = shifted_control.flatten()
        else:
            # 方案 B：冷启动 (Cold Start) - 仅在第一步或上一步求解失败时调用
            # initial_control = self._get_feasible_initial_control(current_state, target_position)
            initial_control = np.zeros(self.horizon * self.control_dim)
        
        # 2. 控制量边界
        control_bounds = []
        for i in range(self.horizon):
            for j in range(self.control_dim):
                if j == 1: # aw
                    control_bounds.append((-self.max_yaw_acceleration, self.max_yaw_acceleration))
                else:      # ax, az
                    control_bounds.append((-self.max_acceleration, self.max_acceleration))
        
        def velocity_constraint(control_sequence):
            control_sequence = control_sequence.reshape(self.horizon, self.control_dim)
            predicted_states = self.predict_trajectory(current_state, control_sequence)
            constraints_list = []
 
            vel_margin = 0.01
            yaw_margin = 0.01
            max_v_sq = (self.max_velocity * (1 - vel_margin)) ** 2
            max_w_sq = (self.max_yaw_rate * (1 - yaw_margin)) ** 2
            
            for i in range(self.horizon + 1):
                vx = float(predicted_states[i][4])
                wz = float(predicted_states[i][5])
                vz = float(predicted_states[i][6])

                constraints_list.append(float(max_v_sq - (vx**2 + vz**2)))
                constraints_list.append(float(max_w_sq - (wz**2)))
                constraints_list.append(float(vx + self.max_velocity))
 
            # # 约束4：障碍物硬约束
            # obs_margin = 0.5
            
            # # === [新增调试1]：用于记录本次内部推演的最危险距离 ===
            # min_test_dist = float('inf') 
            # danger_obs_name = ""

            # if obstacles:
            #     for i in range(self.horizon + 1):
            #         px = float(predicted_states[i][0])
            #         py = float(predicted_states[i][1])
            #         pz = float(predicted_states[i][2])
            #         for obs in obstacles:
            #             c = obs.get("center")
            #             if c is None:
            #                 continue
            #             name = obs.get("name", "obstacle")
                        
            #             if name == "floor":
            #                 r_obs = float(obs.get("radius", 0.5))
            #                 min_dist_z = r_obs + obs_margin
            #                 dist_z = abs(pz - float(c[2]))
            #                 constraints_list.append(float(dist_z - min_dist_z))
            #             else:
            #                 r_obs = float(obs.get("radius", 0.6))
            #                 r_safe = float(obs.get("safe_radius", 0.5))
            #                 min_dist = r_obs + obs_margin + r_safe
            #                 dx = px - float(c[0])
            #                 dy = py - float(c[1])
            #                 dz = pz - float(c[2])
            #                 dist_sq = dx**2 + dy**2 + dz**2
            #                 constraints_list.append(float(dist_sq - min_dist**2))
                            
            #                 # --- 记录三维欧氏距离 ---
            #                 actual_dist = math.sqrt(dist_sq)
            #                 if actual_dist < min_test_dist:
            #                     min_test_dist = actual_dist
            #                     danger_obs_name = name

            return constraints_list


        # 优化
        result = minimize(
            fun=self.cost_function,
            x0=initial_control,
            args=(current_state, target_position, obstacles),
            method='SLSQP',
            bounds=control_bounds,
            constraints={'type': 'ineq', 'fun': velocity_constraint},
            options={'maxiter': 50, 'ftol': 5e-2, 'disp': False}
        )
        
        if not result.success:
            # print(f"失败原因: {result.message}")
            # print(f"状态码: {result.status}")
            # print(f"迭代次数: {getattr(result, 'nit', 'N/A')}")
            # print(f"函数评估次数: {getattr(result, 'nfev', 'N/A')}")
            
            # # 检查约束违反情况
            # try:
            #     initial_constraints = np.array(velocity_constraint(initial_control))
            #     print(f"初始约束值 (应全部>=0): {initial_constraints.tolist()}")
            #     print(f"初始约束违反数: {np.sum(initial_constraints < 0)}")
            #     print(f"最大约束违反: {np.min(initial_constraints):.6f}")
            # except:
            #     print("无法评估初始约束")
            
            # # 检查控制边界违反
            # control_violations = []
            # for i, bound in enumerate(control_bounds):
            #     val = initial_control[i]
            #     if val < bound[0] or val > bound[1]:
            #         control_violations.append((i, val, bound))
            
            # if control_violations:
            #     print(f"控制边界违反: {len(control_violations)} 处")
            #     for idx, val, bound in control_violations[:3]:  # 只显示前3个
            #         print(f"  控制量 {idx}: {val:.3f} 不在 {bound} 内")
            
            # # 检查目标位置和当前状态
            # print(f"当前状态: x={current_state[0]:.2f}, y={current_state[1]:.2f}, z={current_state[2]:.2f}")
            # print(f"目标位置: x={target_position[0]:.2f}, y={target_position[1]:.2f}, z={target_position[2]:.2f}")
            # print(f"距离目标: {np.linalg.norm(current_state[:3] - target_position):.2f}")
            
            # print("=" * 40)
            return None, None, False
        
        # 提取最优控制序列
        optimal_control = result.x.reshape(self.horizon, self.control_dim)
        
        # 预测状态序列
        predicted_states = self.predict_trajectory(current_state, optimal_control)
        
        return optimal_control, predicted_states, True
    
    def get_velocity_command(self, current_state, target_position, obstacles=None):
        """
        获取加速度指令（返回最优控制序列）
        
        参数:
        current_state: 当前状态 [x, y, z, yaw, vx, wz, vz]
        target_position: 目标位置 [x, y, z]
        obstacles: 障碍物列表（可选）
        
        返回:
        accel_commands: 加速度指令序列，形状为 (horizon, 3)，每行为 [ax, aw, az]
        """
        # 求解MPC（无参考轨迹）
        optimal_control, predicted_states, flag = self.solve_mpc(current_state, target_position, obstacles)
        
        if not flag or predicted_states is None:
            # 失败时由上层逻辑决定如何退化处理（例如使用上一次指令的下一个）
            self.control_sequence = None

            return None, False
        
        self.control_sequence = np.array(optimal_control)

        # 返回最优控制序列（加速度）[ax, aw, az]
        return optimal_control, True
    
    
    def update_weights(self, Q=None, R=None, Qf=None, horizon=None, weight=None):
        min_control_weight = 1e-6
        for i in range(len(weight)):
            if weight[i] <= 0:
                weight[i] = min_control_weight
        # weight = [1]*7
        """更新权重矩阵"""
        self.Q = [float(weight[0]), float(weight[1]), float(weight[2]),float(weight[3]),float(weight[4]),float(weight[5]),float(weight[6])]
        self.Qp = np.diag([float(weight[0]), float(weight[1]), float(weight[2])])
        self.R = np.diag([float(weight[3]), float(weight[4]), float(weight[5])])
        if Qf is not None:
            self.Qf = 2*self.Q
        if horizon is not None:
            self.horizon = horizon

    def _get_feasible_initial_control(self, current_state, target_position):
        initial_sequence = np.zeros((self.horizon, self.control_dim))
        
        current_pos = current_state[:3]
        target_yaw = np.arctan2(target_position[1] - current_pos[1], target_position[0] - current_pos[0])
        current_yaw = current_state[3]
        yaw_error = self._wrap_to_pi(target_yaw - current_yaw)
        
        # --- 核心改进：获取当前真实速度 ---
        current_vx = current_state[4]
        current_vz = current_state[6]
        current_speed = np.sqrt(current_vx**2 + current_vz**2)
        
        # --- 智能暗示逻辑 ---
        aw_hint = np.clip(yaw_error, -0.2, 0.2)
        
        # 预留 0.05 的安全余量。如果已经快超速了，坚决不再给正向油门！
        if current_speed >= self.max_velocity - 0.05:
            ax_hint = 0.0
            az_hint = 0.0
        else:
            ax_hint = 0.1
            z_diff = target_position[2] - current_pos[2]
            az_hint = np.clip(z_diff * 0.3, -0.5, 0.5)
        
        # 生成衰减的控制序列
        for i in range(self.horizon):
            initial_sequence[i, 0] = ax_hint * (1.0 - i / self.horizon)
            initial_sequence[i, 1] = aw_hint * (1.0 - i / self.horizon)
            initial_sequence[i, 2] = az_hint * (1.0 - i / self.horizon)
            
        return initial_sequence.flatten()

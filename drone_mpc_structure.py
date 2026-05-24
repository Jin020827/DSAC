import math

import numpy as np
from scipy.optimize import minimize


class DroneMpcStructure:
    """MPC controller copied and trimmed from the main drone project.

    State: [x, y, z, yaw, vx, wz, vz]
    Control: [ax, aw, az]
    """

    def __init__(
        self,
        dt=0.2,
        horizon=15,
        max_velocity=0.5,
        max_acceleration=1.0,
        max_yaw_acceleration=1.0,
        max_yaw_rate=1.0,
        accel_scale=None,
        yaw_accel_scale=None,
    ):
        self.dt = dt
        self.horizon = horizon
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_yaw_acceleration = max_yaw_acceleration
        self.max_yaw_rate = max_yaw_rate
        self.accel_scale = float(accel_scale) if accel_scale is not None else float(max_acceleration)
        self.yaw_accel_scale = (
            float(yaw_accel_scale) if yaw_accel_scale is not None else float(max_yaw_acceleration)
        )

        self.state_dim = 7
        self.control_dim = 3
        self.control_sequence = None
        self.Q = None
        self.R = None
        self.Qp = None

    def state_dynamics(self, state, control):
        x, y, z, yaw, vx, wz, vz = state
        ax, aw, az = control

        vx_new = vx + ax * self.dt
        vz_new = vz + az * self.dt
        wz_new = wz + aw * self.dt
        yaw_new = yaw + wz * self.dt + 0.5 * aw * self.dt**2

        yaw_avg = 0.5 * (yaw + yaw_new)
        x_new = x + (np.cos(yaw_avg) * vx) * self.dt + (np.cos(yaw_avg) * 0.5 * ax * self.dt**2)
        y_new = y + (np.sin(yaw_avg) * vx) * self.dt + (np.sin(yaw_avg) * 0.5 * ax * self.dt**2)
        z_new = z + vz * self.dt + 0.5 * az * self.dt**2

        return np.array([x_new, y_new, z_new, yaw_new, vx_new, wz_new, vz_new])

    def predict_trajectory(self, initial_state, control_sequence):
        states = np.zeros((self.horizon + 1, self.state_dim))
        states[0] = initial_state
        for i in range(self.horizon):
            states[i + 1] = self.state_dynamics(states[i], control_sequence[i])
        return states

    def cost_function(self, control_sequence, initial_state, goal_position, obstacles=None):
        control_sequence = control_sequence.reshape(self.horizon, self.control_dim)
        predicted_states = self.predict_trajectory(initial_state, control_sequence)

        cost_goal = 0.0
        cost_control = 0.0
        cost_obstacle = 0.0
        cost_yaw = 0.0

        current_pos = initial_state[:3]
        dist_curr_to_goal = np.linalg.norm(current_pos - goal_position)
        dynamic_scale = np.sqrt(max(1.0, dist_curr_to_goal))

        for i in range(self.horizon):
            p = predicted_states[i + 1][:3]
            pos_error_norm = ((p - goal_position) / dynamic_scale).reshape(-1, 1)
            cost_goal += float(pos_error_norm.T @ self.Qp @ pos_error_norm)

        for i in range(self.horizon):
            u_norm = np.array(
                [
                    control_sequence[i, 0] / self.accel_scale,
                    control_sequence[i, 1] / self.yaw_accel_scale,
                    control_sequence[i, 2] / self.accel_scale,
                ]
            )
            cost_control += float(u_norm.T @ self.R @ u_norm)

            if i < self.horizon - 1:
                delta_u_norm = np.array(
                    [
                        (control_sequence[i + 1, 0] - control_sequence[i, 0]) / self.accel_scale,
                        (control_sequence[i + 1, 1] - control_sequence[i, 1]) / self.yaw_accel_scale,
                        (control_sequence[i + 1, 2] - control_sequence[i, 2]) / self.accel_scale,
                    ]
                )
                cost_control += float(delta_u_norm.T @ self.R @ delta_u_norm)

        if obstacles:
            for i in range(self.horizon + 1):
                p = predicted_states[i][:3]
                yaw = predicted_states[i][3]
                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)

                for obs in obstacles:
                    c = obs.get("center")
                    if c is None:
                        continue
                    c = np.array(c, dtype=float)
                    r_safe = float(obs.get("safe_radius", 1.0))
                    r_obs = float(obs.get("radius", 0.5))
                    name = obs.get("name", "obstacle")

                    if obs.get("type") == "dynamic":
                        c = c.copy()
                        c[0] += (-0.1) * i

                    if name == "floor":
                        dist_z = abs(p[2] - c[2]) - r_safe
                        d_eff = max(dist_z, 0.0)
                        cost_obstacle += float(self.Q[6]) * 5.0 / (d_eff + 0.2)
                    else:
                        dx = p[0] - c[0]
                        dy = p[1] - c[1]
                        dx_body = dx * cos_y + dy * sin_y
                        dy_body = -dx * sin_y + dy * cos_y
                        d_soft = r_obs + r_safe
                        s_lat = d_soft**2
                        s_long = s_lat * 4.0
                        denom = (dx_body**2 / s_long) + (dy_body**2 / s_lat) + 0.001
                        cost_obstacle += float(self.Q[6]) / denom

        yaw_threshold = math.radians(120.0)
        for i in range(self.horizon + 1):
            p = predicted_states[i][:3]
            yaw = predicted_states[i][3]
            pos_err = goal_position[:2] - p[:2]
            target_yaw = math.atan2(pos_err[1], pos_err[0])
            yaw_err = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
            if yaw_err > yaw_threshold:
                cost_yaw += 5.0 * ((yaw_err - yaw_threshold) ** 2)

        return float(cost_goal) + float(cost_control) + float(cost_obstacle) + float(cost_yaw)

    def solve_mpc(self, current_state, target_position, obstacles=None):
        if self.control_sequence is not None:
            shifted_control = np.zeros_like(self.control_sequence)
            shifted_control[:-1] = self.control_sequence[1:]
            shifted_control[-1] = np.zeros(self.control_dim)
            initial_control = shifted_control.flatten()
        else:
            initial_control = np.zeros(self.horizon * self.control_dim)

        control_bounds = []
        for _ in range(self.horizon):
            control_bounds.append((-self.max_acceleration, self.max_acceleration))
            control_bounds.append((-self.max_yaw_acceleration, self.max_yaw_acceleration))
            control_bounds.append((-self.max_acceleration, self.max_acceleration))

        def velocity_constraint(control_sequence):
            control_sequence = control_sequence.reshape(self.horizon, self.control_dim)
            predicted_states = self.predict_trajectory(current_state, control_sequence)
            constraints_list = []
            max_v_sq = (self.max_velocity * 0.99) ** 2
            max_w_sq = (self.max_yaw_rate * 0.99) ** 2

            for i in range(self.horizon + 1):
                vx = float(predicted_states[i][4])
                wz = float(predicted_states[i][5])
                vz = float(predicted_states[i][6])
                constraints_list.append(float(max_v_sq - (vx**2 + vz**2)))
                constraints_list.append(float(max_w_sq - (wz**2)))
                constraints_list.append(float(vx + self.max_velocity))

            return constraints_list

        result = minimize(
            fun=self.cost_function,
            x0=initial_control,
            args=(current_state, target_position, obstacles),
            method="SLSQP",
            bounds=control_bounds,
            constraints={"type": "ineq", "fun": velocity_constraint},
            options={"maxiter": 50, "ftol": 5e-2, "disp": False},
        )

        if not result.success:
            self.control_sequence = None
            return None, None, False

        optimal_control = result.x.reshape(self.horizon, self.control_dim)
        predicted_states = self.predict_trajectory(current_state, optimal_control)
        self.control_sequence = np.array(optimal_control)
        return optimal_control, predicted_states, True

    def get_velocity_command(self, current_state, target_position, obstacles=None):
        optimal_control, predicted_states, success = self.solve_mpc(
            current_state, target_position, obstacles
        )
        if not success or predicted_states is None:
            return None, False
        return optimal_control, True

    def update_weights(self, weight):
        min_control_weight = 1e-6
        safe_weight = [max(float(v), min_control_weight) for v in weight]
        self.Q = safe_weight
        self.Qp = np.diag(safe_weight[:3])
        self.R = np.diag(safe_weight[3:6])

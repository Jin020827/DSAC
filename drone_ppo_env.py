import math
import time

import airsim
import numpy as np

from drone_mpc_structure import DroneMpcStructure
from utils import generate_points, global2body


class DronePPOEnv:
    """Gym-like PPO environment for event-triggered drone MPC.

    action=1 solves a fresh MPC sequence.
    action=0 reuses the next control from the previous MPC sequence.
    """

    def __init__(
        self,
        index=0,
        map_index=102,
        max_timesteps=200,
        dt=0.2,
        reward_mode="task",
        obs_mode="task",
        fixed_points=True,
        rho=0.1,
        trigger_steps_before_end=9,
        show_predicted_trajectory=False,
        settle_time=0.2,
    ):
        if reward_mode not in ("task", "mpc_cost"):
            raise ValueError("reward_mode must be 'task' or 'mpc_cost'")
        if obs_mode not in ("task", "origin_like"):
            raise ValueError("obs_mode must be 'task' or 'origin_like'")

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.index = index
        self.vehicle_name = "Drone_" + str(index)
        self.map_index = map_index
        self.max_timesteps = max_timesteps
        self.dt = dt
        self.reward_mode = reward_mode
        self.obs_mode = obs_mode
        self.fixed_points = fixed_points
        self.rho = rho
        self.trigger_steps_before_end = trigger_steps_before_end
        self.show_predicted_trajectory = show_predicted_trajectory
        self.settle_time = settle_time

        self.action_space = 2
        self.obs_space = 34 if obs_mode == "task" else 14

        self.goal = np.array([10.0, 10.0, -2.5], dtype=float)
        self.all_obstacles = []
        self.obstacles = []
        self.dynamic_origin_cache = {}
        self.dynamic_positions = {}
        self.record_pos = []
        self.current_control = [0.0, 0.0, 0.0]
        self.pre_control = [0.0, 0.0, 0.0]

        self.fixed_q = [0.5, 0.5, 3.0, 0.5, 0.5, 0.5, 0.5]
        self.mpc = DroneMpcStructure(dt=dt)
        self.mpc.update_weights(self.fixed_q)

        self.last_mpc_control_sequence = None
        self.last_predicted_states = None
        self.last_mpc_index = 0
        self.step_count = 0

    def reset(self):
        self.client.simPause(False)
        self.reset_world()
        time.sleep(0.5)
        pose_list, goal_list = generate_points(
            num_env=max(1, self.index + 1), map_index=self.map_index, fixed=self.fixed_points
        )
        start = pose_list[self.index]
        self.goal = np.array(goal_list[self.index], dtype=float)

        temp_pose = airsim.Pose()
        temp_pose.position.x_val = float(start[0])
        temp_pose.position.y_val = float(start[1])
        temp_pose.position.z_val = float(start[2])
        self.client.simSetVehiclePose(temp_pose, ignore_collision=True, vehicle_name=self.vehicle_name)
        if self.settle_time > 0:
            time.sleep(self.settle_time)

        self.drones_init()
        self.reset_pose(start)
        time.sleep(0.5)
        self.get_obstacles()

        self.last_mpc_control_sequence = None
        self.last_predicted_states = None
        self.last_mpc_index = 0
        self.step_count = 0
        self.record_pos = []
        self.current_control = [0.0, 0.0, 0.0]
        self.pre_control = [0.0, 0.0, 0.0]
        self.mpc.control_sequence = None

        return self.get_observation()

    def step(self, action):
        self.step_count += 1
        self.move_obstacles(self.step_count)
        self.get_obstacle_vector()

        current_mpc_state = self.convert_env_state_to_mpc_state()
        rule_triggered = self._must_trigger(current_mpc_state)
        agent_wanted_update = int(action) > 0
        should_update_mpc = agent_wanted_update or rule_triggered
        mpc_success = None
        reused_control = False
        reused_mpc_index = None
        control_source = "mpc_update" if should_update_mpc else "reuse"

        if should_update_mpc:
            mpc_control_sequence, success = self.mpc.get_velocity_command(
                current_mpc_state, self.goal, obstacles=self.obstacles
            )
            mpc_success = bool(success and mpc_control_sequence is not None)
            if success and mpc_control_sequence is not None:
                self.last_mpc_control_sequence = mpc_control_sequence
                self.last_predicted_states = self.mpc.predict_trajectory(current_mpc_state, mpc_control_sequence)
                self.last_mpc_index = 0
                mpc_accel = mpc_control_sequence[0]
            else:
                mpc_accel = self._fallback_accel(current_mpc_state)
                self.last_mpc_control_sequence = None
                self.last_predicted_states = None
        else:
            if (
                self.last_mpc_control_sequence is not None
                and self.last_mpc_index < len(self.last_mpc_control_sequence) - 1
            ):
                self.last_mpc_index += 1
                mpc_accel = self.last_mpc_control_sequence[self.last_mpc_index]
                reused_control = True
                reused_mpc_index = self.last_mpc_index
            else:
                rule_triggered = True
                mpc_success = False
                mpc_accel = self._fallback_accel(current_mpc_state)
                control_source = "fallback_no_sequence"

        velocity = self._accel_to_velocity(mpc_accel)
        self.control_vel(velocity)

        if self.show_predicted_trajectory and self.last_predicted_states is not None:
            self.plot_predicted_trajectory(self.last_predicted_states, duration=0.1)

        if self.dt > 0:
            time.sleep(self.dt)

        is_crashed = self.get_crash_state()
        reward_task, done, result = self._task_reward(
            update_flag=agent_wanted_update,
            rule_triggered=rule_triggered,
            is_crashed=is_crashed,
        )
        reward_cost, jmpc = self._mpc_cost_reward(agent_wanted_update)
        reward = reward_task if self.reward_mode == "task" else reward_cost

        self.record_pos.append(self.get_state())
        obs = self.get_observation()
        info = {
            "time": self.step_count * self.dt,
            "jmpc": jmpc,
            "task_reward": reward_task,
            "mpc_cost_reward": reward_cost,
            "result": result,
            "triggered": int(should_update_mpc),
            "mpc_success": mpc_success,
            "agent_action": int(agent_wanted_update),
            "rule_triggered": int(rule_triggered),
            "reused_control": int(reused_control),
            "mpc_index": reused_mpc_index,
            "control": np.asarray(mpc_accel, dtype=float).tolist(),
            "velocity": [float(v) for v in velocity],
            "control_source": control_source,
        }
        return obs, float(reward), bool(done), info

    def close(self):
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass

    def _must_trigger(self, current_mpc_state):
        if self.last_mpc_control_sequence is None:
            return True
        remaining = len(self.last_mpc_control_sequence) - self.last_mpc_index
        if remaining <= self.trigger_steps_before_end:
            return True
        current_speed = np.sqrt(current_mpc_state[4] ** 2 + current_mpc_state[6] ** 2)
        return current_speed < 0.3

    def _accel_to_velocity(self, mpc_accel):
        current_mpc_state = self.convert_env_state_to_mpc_state()
        new_vx = current_mpc_state[4] + float(mpc_accel[0]) * self.dt
        new_wz = current_mpc_state[5] + float(mpc_accel[1]) * self.dt
        new_vz = current_mpc_state[6] + float(mpc_accel[2]) * self.dt

        velocity_magnitude = np.sqrt(new_vx**2 + new_vz**2)
        if velocity_magnitude > self.mpc.max_velocity:
            scale = self.mpc.max_velocity / velocity_magnitude
            new_vx *= scale
            new_vz *= scale
        if abs(new_wz) > self.mpc.max_yaw_rate:
            new_wz = np.sign(new_wz) * self.mpc.max_yaw_rate
        return [new_vx, new_wz, new_vz]

    def _fallback_accel(self, current_mpc_state):
        pos_error = self.goal - current_mpc_state[:3]
        distance = np.linalg.norm(pos_error)
        target_yaw = np.arctan2(pos_error[1], pos_error[0])
        yaw_error = np.arctan2(
            np.sin(target_yaw - current_mpc_state[3]), np.cos(target_yaw - current_mpc_state[3])
        )

        desired_speed = min(1.0, max(0.1, distance / 3.0))
        if abs(yaw_error) > np.pi / 4:
            desired_speed *= 0.3
        desired_vz = np.clip(pos_error[2] * 0.5, -0.5, 0.5)
        desired_wz = np.clip(yaw_error * 0.8, -1.0, 1.0)

        speed_error = desired_speed - current_mpc_state[4]
        ax = np.clip(speed_error * 0.5, -0.5, 0.5)
        aw = desired_wz
        az = np.clip((desired_vz - current_mpc_state[6]) / self.dt, -0.5, 0.5)
        return np.array([ax, aw, az], dtype=float)

    def _mpc_cost_reward(self, agent_wanted_update):
        current_mpc_state = self.convert_env_state_to_mpc_state()
        if self.last_mpc_control_sequence is None:
            jmpc = self.mpc.cost_function(
                np.zeros((self.mpc.horizon, self.mpc.control_dim)),
                current_mpc_state,
                self.goal,
                self.obstacles,
            )
        else:
            sequence = np.zeros_like(self.last_mpc_control_sequence)
            remaining = len(self.last_mpc_control_sequence) - self.last_mpc_index
            sequence[:remaining] = self.last_mpc_control_sequence[self.last_mpc_index :]
            jmpc = self.mpc.cost_function(sequence, current_mpc_state, self.goal, self.obstacles)
        reward = -float(jmpc) * self.dt - self.rho * float(agent_wanted_update)
        return reward, float(jmpc)

    def _task_reward(self, update_flag=False, rule_triggered=False, is_crashed=False):
        current_pos = np.array(self.get_state(return_index=1))
        dist_curr = np.linalg.norm(current_pos - self.goal)
        local_goal_vec, _ = self.get_local_goal_and_speed()
        yaw_err = abs(math.atan2(local_goal_vec[1], local_goal_vec[0]))
        yaw_err = -(yaw_err / math.pi)

        done = False
        r_event = 0.0
        result = "Running"
        if dist_curr < 2.5:
            r_event = 20.0
            done = True
            result = "Reach Goal"
        elif is_crashed:
            r_event = -10.0
            done = True
            result = "Crashed"
        elif self.step_count >= self.max_timesteps:
            r_event = -10.0
            done = True
            result = "Time out"

        _, risk_relu_list, _ = self.calculate_dynamic_risk()
        safety_mpc = sum(risk_relu_list) if risk_relu_list else 0.0
        r_safe = -max(0.0, -safety_mpc) - max(0.0, -yaw_err)
        r_yaw = -max(0.0, -yaw_err)

        if update_flag:
            r_agent = -0.1
        elif rule_triggered:
            r_agent = -0.5
        elif risk_relu_list:
            r_agent = r_safe
        else:
            r_agent = r_yaw

        self.pre_control = list(self.current_control)
        return float(r_agent + r_event), done, result

    def get_observation(self):
        current_mpc_state = self.convert_env_state_to_mpc_state()
        if self.obs_mode == "task":
            obstacle_vector = np.array(self.get_obstacle_vector(), dtype=np.float32)
            yaw = np.array([self.get_state(return_index=6)], dtype=np.float32)
            velocity = np.array(current_mpc_state[4:], dtype=np.float32)
            return np.concatenate([obstacle_vector, yaw, velocity]).astype(np.float32)

        if (
            self.last_predicted_states is not None
            and self.last_mpc_index < len(self.last_predicted_states)
        ):
            predicted = self.last_predicted_states[self.last_mpc_index]
        else:
            predicted = np.zeros(7, dtype=np.float32)
        return np.concatenate([current_mpc_state, predicted]).astype(np.float32)

    def convert_env_state_to_mpc_state(self):
        position = self.get_state(return_index=1)
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        angular_vel = state.kinematics_estimated.angular_velocity
        _, local_speed = self.get_local_goal_and_speed()
        _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        global_vz = state.kinematics_estimated.linear_velocity.z_val
        return np.array(
            [
                position[0],
                position[1],
                position[2],
                yaw,
                local_speed[0],
                angular_vel.z_val,
                global_vz,
            ],
            dtype=float,
        )

    def reset_world(self):
        if self.index == 0:
            self.client.reset()
        self.all_obstacles = []
        self.obstacles = []
        for obj_name, init_pos in self.dynamic_origin_cache.items():
            self.dynamic_positions[obj_name] = init_pos.copy()
            try:
                orig_state = self.client.simGetObjectPose(object_name=obj_name)
                reset_pose = airsim.Pose(
                    position_val=airsim.Vector3r(float(init_pos[0]), float(init_pos[1]), float(init_pos[2])),
                    orientation_val=orig_state.orientation,
                )
                self.client.simSetObjectPose(object_name=obj_name, pose=reset_pose, teleport=True)
            except Exception:
                continue

    def drones_init(self):
        self.client.simFlushPersistentMarkers()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

    def reset_pose(self, start):
        pose = airsim.Pose()
        pose.position.x_val = float(start[0])
        pose.position.y_val = float(start[1])
        pose.position.z_val = float(start[2])
        dx = self.goal[0] - start[0]
        dy = self.goal[1] - start[1]
        pose.orientation = airsim.to_quaternion(0, 0, math.atan2(dy, dx))
        self.client.simSetVehiclePose(vehicle_name=self.vehicle_name, ignore_collision=False, pose=pose)

    def get_state(self, return_index=1):
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        position = pose.position
        if return_index == 1:
            return [position.x_val, position.y_val, position.z_val]
        if return_index == 5:
            _, _, yaw = airsim.to_eularian_angles(pose.orientation)
            return [position.x_val, position.y_val, position.z_val, math.degrees(yaw), math.sin(yaw), math.cos(yaw)]
        if return_index == 6:
            local_goal_vec, _ = self.get_local_goal_and_speed()
            yaw = abs(math.atan2(local_goal_vec[1], local_goal_vec[0]))
            return -(yaw / math.pi)
        return [position.x_val, position.y_val, position.z_val]

    def get_local_goal_and_speed(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        roll, pitch, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        pos = np.asarray(self.get_state(return_index=1), dtype=float)
        local_goal = global2body(roll, pitch, yaw, np.asarray(self.goal, dtype=float), pos)
        velocity = state.kinematics_estimated.linear_velocity
        v_xyz = np.array([velocity.x_val, velocity.y_val, velocity.z_val], dtype=float)
        local_v = global2body(roll, pitch, yaw, v_xyz, np.array([0.0, 0.0, 0.0]))
        angular_v = state.kinematics_estimated.angular_velocity
        return local_goal, np.asarray([local_v[0], local_v[2], angular_v.z_val], dtype=float)

    def control_vel(self, cmd):
        vx_body, wz, vz_cmd = cmd
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)
        vx_global = float(vx_body) * math.cos(yaw)
        vy_global = float(vx_body) * math.sin(yaw)
        vz_real = float(np.clip(vz_cmd, -0.5, 0.5))
        self.client.moveByVelocityAsync(
            vx=vx_global,
            vy=vy_global,
            vz=vz_real,
            duration=1.5,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(wz) * 180.0 / np.pi),
            vehicle_name=self.vehicle_name,
        )
        self.current_control = [float(vx_body), float(wz), vz_real]

    def get_obstacle_vector(self):
        max_dist = 6.0
        fov_cos_threshold = 0.707
        max_obs = 5
        feature_dim = 6
        final_vector = np.zeros(max_obs * feature_dim, dtype=np.float32)
        if not self.all_obstacles:
            self.get_obstacles()

        uav_state = self.get_state(return_index=5)
        uav_pos = np.array(uav_state[:3], dtype=float)
        uav_yaw_rad = math.radians(uav_state[3])
        forward_vec = np.array([math.cos(uav_yaw_rad), math.sin(uav_yaw_rad)])
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        roll, pitch, yaw = airsim.to_eularian_angles(pose.orientation)

        obs_list = []
        for obstacle in self.all_obstacles:
            obs_center = obstacle["center"].copy()
            if np.isnan(obs_center).any():
                continue
            obs_center[2] = uav_pos[2]
            rel_pos_global = obs_center[:2] - uav_pos[:2]
            dist = np.linalg.norm(rel_pos_global)
            if np.isnan(dist) or dist < 1e-3 or dist >= max_dist:
                continue
            rel_pos_unit = rel_pos_global / dist
            if np.dot(forward_vec, rel_pos_unit) <= fov_cos_threshold:
                continue
            local_pos = global2body(roll, pitch, yaw, obs_center, uav_pos)
            angle = math.atan2(local_pos[1], local_pos[0])
            obs_feature = [
                local_pos[0],
                local_pos[1],
                local_pos[2],
                np.clip(dist / max_dist, 0.0, 1.0),
                math.sin(angle),
                math.cos(angle),
            ]
            if not np.isnan(obs_feature).any():
                obs_list.append((dist, obs_feature, obstacle))

        obs_list.sort(key=lambda x: x[0])
        self.obstacles = [item[2] for item in obs_list]
        for i in range(min(len(obs_list), max_obs)):
            final_vector[i * feature_dim : (i + 1) * feature_dim] = obs_list[i][1]
        return final_vector

    def calculate_dynamic_risk(self):
        risk_tanh_list = []
        risk_relu_list = []
        distance_list = []
        current_pos = np.array(self.get_state(return_index=1), dtype=float)
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        vel = state.kinematics_estimated.linear_velocity
        v_vec_2d = np.array([vel.x_val, vel.y_val], dtype=float)
        v_norm_2d = np.linalg.norm(v_vec_2d)
        if v_norm_2d < 0.01:
            return risk_tanh_list, risk_relu_list, distance_list

        v_hat_2d = v_vec_2d / v_norm_2d
        r_drone = 0.6
        for obs in self.obstacles:
            obs_pos = obs["center"]
            danger_threshold = obs.get("radius", 0.5) + r_drone + obs.get("safe_radius", 0.5)
            d_vec_2d = obs_pos[:2] - current_pos[:2]
            d_norm_2d = np.linalg.norm(d_vec_2d)
            if d_norm_2d < 1e-3 or np.dot(d_vec_2d, v_hat_2d) < 0.0:
                continue
            d_lat = abs(np.cross(d_vec_2d, v_hat_2d))
            lat_gap = d_lat - danger_threshold
            risk_tanh = float(np.tanh(lat_gap)) if lat_gap >= 0 else min(lat_gap / danger_threshold, 0.0)
            risk_relu = min(lat_gap / danger_threshold, 0.0)
            risk_tanh_list.append(risk_tanh)
            risk_relu_list.append(risk_relu)
            distance_list.append(float(d_norm_2d))
        return risk_tanh_list, risk_relu_list, distance_list

    def get_crash_state(self):
        try:
            collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        except Exception:
            collision = False
        return collision or self._check_physical_collision(np.array(self.get_state(return_index=1), dtype=float))

    def _check_physical_collision(self, drone_pos):
        if not self.all_obstacles:
            return False
        drone_pos_xy = drone_pos[:2]
        for obs in self.all_obstacles:
            if obs.get("type") == "dynamic":
                continue
            dist_xy = np.linalg.norm(drone_pos_xy - obs["center"][:2])
            if dist_xy < (0.4 + obs.get("radius", 0.5) + 0.1):
                return True
        return False

    def get_obstacles(self):
        map_configs = {
            101: {"static": {"num": 100, "r": 0.5, "safe": 0.5}, "dynamic": {"num": 0, "r": 0.0, "safe": 1.2}},
            102: {"static": {"num": 100, "r": 0.5, "safe": 0.5}, "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}},
            103: {"static": {"num": 100, "r": 0.5, "safe": 0.5}, "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}},
            104: {"static": {"num": 100, "r": 0.5, "safe": 0.5}, "dynamic": {"num": 0, "r": 0.0, "safe": 0.0}},
            105: {"static": {"num": 100, "r": 0.5, "safe": 0.5}, "dynamic": {"num": 10, "r": 0.5, "safe": 0.8}},
        }
        cfg = map_configs.get(self.map_index)
        self.all_obstacles = []
        if not cfg:
            return
        for i in range(1, cfg["static"]["num"] + 1):
            try:
                name = "Obstacle_" + str(i)
                pose = self.client.simGetObjectPose(object_name=name)
                pos = pose.position
                if not np.isnan(pos.x_val):
                    self.all_obstacles.append(
                        {
                            "center": np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)]),
                            "radius": cfg["static"]["r"],
                            "safe_radius": cfg["static"]["safe"],
                            "type": "static",
                            "name": name,
                        }
                    )
            except Exception:
                continue

        for i in range(1, cfg["dynamic"]["num"] + 1):
            try:
                name = "D_Obstacle_" + str(i)
                pose = self.client.simGetObjectPose(object_name=name)
                pos = pose.position
                origin_pos = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
                self.dynamic_origin_cache.setdefault(name, origin_pos.copy())
                self.dynamic_positions.setdefault(name, origin_pos.copy())
                self.all_obstacles.append(
                    {
                        "center": origin_pos.copy(),
                        "radius": cfg["dynamic"]["r"],
                        "safe_radius": cfg["dynamic"]["safe"],
                        "type": "dynamic",
                        "name": name,
                    }
                )
            except Exception:
                continue

    def move_obstacles(self, step):
        map_configs = {
            11: {"dynamic": {"num": 3}},
            12: {"dynamic": {"num": 3}},
            101: {"dynamic": {"num": 0}},
            102: {"dynamic": {"num": 0}},
            103: {"dynamic": {"num": 0}},
            104: {"dynamic": {"num": 0}},
            105: {"dynamic": {"num": 10}},
        }
        cfg = map_configs.get(self.map_index)
        if not cfg or cfg["dynamic"]["num"] == 0:
            return
        for i in range(1, cfg["dynamic"]["num"] + 1):
            name = "D_Obstacle_" + str(i)
            try:
                if name not in self.dynamic_positions:
                    pose = self.client.simGetObjectPose(object_name=name)
                    pos = pose.position
                    self.dynamic_positions[name] = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
                new_pos = self.dynamic_positions[name].copy()
                new_pos[0] += -0.4
                self.dynamic_positions[name] = new_pos
                for obs in self.all_obstacles:
                    if obs.get("name") == name:
                        obs["center"] = new_pos.copy()
                        break
                orig_pose = self.client.simGetObjectPose(object_name=name)
                new_pose = airsim.Pose(
                    position_val=airsim.Vector3r(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])),
                    orientation_val=orig_pose.orientation,
                )
                self.client.simSetObjectPose(object_name=name, pose=new_pose, teleport=True)
            except Exception:
                continue

    def plot_predicted_trajectory(self, predicted_states, color_rgba=None, thickness=8, duration=0.2):
        if color_rgba is None:
            color_rgba = [0.0, 1.0, 0.0, 1.0]
        if predicted_states is None or len(predicted_states) < 2:
            return
        points = [
            airsim.Vector3r(float(state[0]), float(state[1]), float(state[2]))
            for state in predicted_states
        ]
        for i in range(len(points) - 1):
            self.client.simPlotLineList(
                [points[i], points[i + 1]], color_rgba, thickness, is_persistent=False, duration=duration
            )

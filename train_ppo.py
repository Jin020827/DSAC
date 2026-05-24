import argparse
import csv
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# 如果你希望直接在代码中设置训练/测试参数，打开此开关。
# 否则可以通过终端命令行参数来控制运行。
USE_CODE_CONFIG = True

# 训练/测试模式："train" 或 "test"
# CODE_PHASE = "train"
CODE_PHASE = "test"

# 奖励模式："task" 或 "mpc_cost"
CODE_REWARD_MODE = "task"

# 观测模式："task" 或 "origin_like"
CODE_OBS_MODE = "task"

# 地图索引，只影响当前运行使用的 AirSim 地图
CODE_MAP_INDEX = 102

# 多机环境中的 agent 索引，通常为 0
CODE_INDEX = 0

# 训练总轮数，仅在 train 模式下有效
CODE_TOTAL_EPISODES = 1000

# 测试轮数，仅在 test 模式下有效
CODE_TEST_EPISODES = 30

# 是否加载已保存模型继续训练或测试
# CODE_LOAD_MODEL = False
CODE_LOAD_MODEL = True

# 要加载的 runs 子目录名，例如：
# "PPO_drone_task_task_2026-05-18-17-48-27"
CODE_LOAD_RUN_NAME = "PPO_drone_task_task_2026-05-19-17-21-34"

# 要加载的模型轮数；填整数加载 ppo_ep_<N>.pt，填 "final" 加载 ppo_final.pt
CODE_LOAD_EPISODE = "final"

# 继续训练开关：仅在 train 且 CODE_LOAD_MODEL=True 时有效
CODE_RESUME = True

# 是否使用 GPU
CODE_CUDA = True

# 是否显示预测轨迹，不常用
CODE_SHOW_PREDICTED_TRAJECTORY = True

# 提前触发步骤数
CODE_TRIGGER_STEPS_BEFORE_END = 9

# 是否使用固定初始点
CODE_FIXED_POINTS = True

# 稳定时间参数
CODE_SETTLE_TIME = 0.2

# 是否在终端打印每个回合结束后的摘要信息
CODE_DISP_INFO_FLAG = True

# 是否在终端打印每一步的详细动作信息，通常保持 False
CODE_STEP_INFO_FLAG = True

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
PROJECT_CODE_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from Logger import Logger  # noqa: E402
from utils import Statistics  # noqa: E402

from drone_ppo_env import DronePPOEnv
from PPO.config import AgentConfig
from PPO.network import MlpPolicy


def get_mpi_rank():
    if MPI is not None:
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
    return rank, size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model_path(run_name, episode):
    if not run_name:
        raise ValueError("CODE_LOAD_RUN_NAME/--load_run_name is required when loading a model")
    model_file = "ppo_final.pt" if str(episode).lower() == "final" else "ppo_ep_{}.pt".format(int(episode))
    return str(RUNS_DIR / run_name / "save_model" / model_file)


def normalize_model_path(path):
    model_path = Path(path).expanduser()
    if model_path.is_absolute():
        return model_path

    cwd_path = (Path.cwd() / model_path).resolve()
    if cwd_path.exists():
        return cwd_path

    if model_path.parts and model_path.parts[0] == "runs":
        return (SCRIPT_DIR / model_path).resolve()
    return cwd_path


def infer_run_dir_from_model_path(model_path):
    model_path = normalize_model_path(model_path)
    if model_path.parent.name == "save_model":
        return model_path.parent.parent
    return model_path.parent


def build_run_dir(args):
    if args.phase == "train" and args.load_model is not None and args.resume:
        return str(infer_run_dir_from_model_path(args.load_model)), True

    run_prefix = "PPO_drone_test" if args.phase == "test" else "PPO_drone"
    return str(
        RUNS_DIR / "{}_{}_{}_{}".format(
            run_prefix,
            args.reward_mode,
            args.obs_mode,
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )
    ), False


def emit_startup_info(agent, args, run_dir, append_train_log):
    messages = [
        "运行模式: {}".format("训练" if args.phase == "train" else "测试"),
        "模型保存目录: {}".format(run_dir),
    ]
    if args.phase == "train":
        messages.insert(0, "训练模式：开始训练 PPO 模型")

    if args.load_model is None:
        messages.append("模型加载: 未加载历史模型，已创建新的 run 目录")
    elif args.phase == "train" and args.resume:
        messages.extend(
            [
                "模型加载: 已加载历史模型并断点续训",
                "加载模型路径: {}".format(args.load_model),
                "续训起始: checkpoint episode={}，下一回合从 {} 开始".format(
                    agent.resume_episode,
                    agent.resume_episode + 1,
                ),
                "日志写入: {}".format("追加到原 train_logs.csv" if append_train_log else "新建 train_logs.csv"),
            ]
        )
    elif args.phase == "train":
        messages.extend(
            [
                "模型加载: 已加载历史模型作为初始化参数，不按原 episode 续训",
                "加载模型路径: {}".format(args.load_model),
                "训练起始: Episode 1，新建 run 目录保存后续模型",
            ]
        )
    else:
        messages.extend(
            [
                "模型加载: 已加载模型用于测试",
                "加载模型路径: {}".format(args.load_model),
                "测试输出目录: {}".format(run_dir),
            ]
        )

    for msg in messages:
        print(msg)
        agent.logger.info(msg)


class Agent(AgentConfig):
    def __init__(self, log_dir, writer, args, device):
        self.env = DronePPOEnv(
            index=args.index,
            map_index=args.map_index,
            max_timesteps=args.max_timesteps,
            dt=args.dt,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            fixed_points=args.fixed_points,
            rho=args.rho,
            trigger_steps_before_end=args.trigger_steps_before_end,
            show_predicted_trajectory=args.show_predicted_trajectory,
            settle_time=args.settle_time,
        )
        self.env_test = DronePPOEnv(
            index=args.index,
            map_index=args.map_index,
            max_timesteps=args.max_timesteps,
            dt=args.dt,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            fixed_points=args.fixed_points,
            rho=args.rho,
            trigger_steps_before_end=args.trigger_steps_before_end,
            show_predicted_trajectory=args.show_predicted_trajectory,
            settle_time=args.settle_time,
        )
        self.action_size = self.env.action_space
        self.obs_size = self.env.obs_space
        self.policy_network = MlpPolicy(
            action_size=self.action_size,
            input_size=self.obs_size,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        ).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.k_epoch, gamma=0.999
        )
        self.loss = 0
        self.writer = writer
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, "save_model")
        self.device = device
        self.disp_info_flag = getattr(args, "disp_info_flag", True)
        self.step_info_flag = getattr(args, "step_info_flag", False)
        self.rollout_memory_size = self.memory_size * self.update_freq
        self.success_list = []
        self.criterion = nn.MSELoss()
        self.rank = getattr(args, "rank", 0)
        self.logger = Logger(
            os.path.join(log_dir, f"ppo_debug_rank{self.rank}.log"),
            clevel=logging.INFO,
            Flevel=logging.INFO,
            CMD_render=False,
            propagate=False,
        )
        self.memory = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "action_prob": [],
            "terminal": [],
            "count": 0,
            "advantage": [],
            "td_target": torch.FloatTensor([]),
        }

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        train_log_path = os.path.join(log_dir, "train_logs.csv")
        append_train_log = getattr(args, "append_train_log", False) and os.path.exists(train_log_path)
        with open(train_log_path, mode="a" if append_train_log else "w", newline="") as csv_file:
            writer_csv = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "Model",
                    "Episode",
                    "RewardMode",
                    "ObsMode",
                    "TrainReturn",
                    "EvalReturn",
                    "EvalSteps",
                    "Success",
                    "ActionFreq",
                    "RuleTriggerFreq",
                    "MpcCostSum",
                ],
            )
            if not append_train_log:
                writer_csv.writeheader()

    def _record_success(self, success):
        if len(self.success_list) == 200:
            self.success_list.pop(0)
        self.success_list.append(1 if success else 0)
        return float(np.sum(self.success_list) / len(self.success_list))

    def _episode_summary(self, env, episode, steps, reward, result, success_rate):
        goal = getattr(env, "goal", [0.0, 0.0, 0.0])
        return (
            "Env %02d, Goal (%05.1f, %05.1f, %05.1f), Episode %05d, "
            "step %03d, Reward %-5.1f, %s, sucessrate:%02.2f"
            % (
                getattr(env, "index", 0),
                goal[0],
                goal[1],
                goal[2],
                episode,
                steps,
                reward,
                result,
                success_rate,
            )
        )

    def _print_episode_result(self, reward, result):
        if self.disp_info_flag:
            print("Reward %-5.1f, Result %s" % (reward, result))

    def _log_step(self, msg):
        self.logger.info(msg)
        if self.step_info_flag:
            print(msg)

    def _format_step_info(self, step, info):
        control = info.get("control", [0.0, 0.0, 0.0])
        control_text = "[{:.3f}, {:.3f}, {:.3f}]".format(
            float(control[0]),
            float(control[1]),
            float(control[2]),
        )
        if info.get("triggered", 0):
            if info.get("mpc_success") is False:
                return "Step %d: 触发MPC计算，MPC优化失败，使用fallback控制 - 控制:%s" % (
                    step,
                    control_text,
                )
            return "Step %d: 触发MPC计算，MPC优化成功 - 控制:%s" % (
                step,
                control_text,
            )
        if info.get("reused_control", 0):
            return "Step %d: 未触发MPC计算，使用上一次控制序列，索引: %s，控制:%s" % (
                step,
                info.get("mpc_index"),
                control_text,
            )
        return "Step %d: 未触发MPC计算，控制序列不可用，使用fallback控制 - 控制:%s" % (
            step,
            control_text,
        )

    def train(self, total_episodes):
        resume_episode = getattr(self, "resume_episode", 0)
        start_episode = resume_episode + 1
        if start_episode > total_episodes:
            msg = (
                "当前加载模型的 episode=%d，total_episodes=%d，没有新的训练回合需要执行"
                % (resume_episode, total_episodes)
            )
            self.logger.info(msg)
            if self.disp_info_flag:
                print(msg)
            self.env.close()
            self.env_test.close()
            return

        global_step = 0
        last_episode = resume_episode
        rollout_episode_count = 0
        for episode in range(start_episode, total_episodes + 1):
            last_episode = episode
            state = self.env.reset()
            total_episode_reward = 0.0
            actions = []
            terminal = False
            episode_length = 0
            info = {"result": "Unknown"}

            while not terminal:
                global_step += 1
                episode_length += 1
                prob_a = self.policy_network.pi(
                    torch.FloatTensor(state).to(self.device)
                )
                action = torch.distributions.Categorical(prob_a).sample().item()
                next_state, reward, terminal, info = self.env.step(action)
                if self.step_info_flag:
                    self._log_step(self._format_step_info(episode_length, info))
                self.add_memory(
                    state, action, reward, next_state, terminal, prob_a[action].item()
                )

                state = next_state
                total_episode_reward += reward
                actions.append(action)

            result = info["result"]
            success_rate = self._record_success(result == "Reach Goal")
            self.logger.info(
                self._episode_summary(
                    self.env,
                    episode,
                    episode_length,
                    total_episode_reward,
                    result,
                    success_rate,
                )
            )
            self._print_episode_result(total_episode_reward, result)

            self.writer.add_scalar("Train/steps", episode_length, episode)
            self.writer.add_scalar("Train/EpisodeReturns", total_episode_reward, episode)
            self.writer.add_scalar("Train/action_ratio", sum(actions) / max(len(actions), 1), episode)
            self.finish_path(episode_length)
            rollout_episode_count += 1

            if rollout_episode_count >= self.update_freq:
                msg = "PPO网络更新：Episode %05d, 累计回合 %d, k_epoch %d, memory_count %d" % (
                    episode,
                    rollout_episode_count,
                    self.k_epoch,
                    self.memory["count"],
                )
                self.logger.info(msg)
                if self.disp_info_flag:
                    print(msg)
                for _ in range(self.k_epoch):
                    self.update_network()
                self.clear_memory()
                rollout_episode_count = 0

            if episode % self.eval_freq == 0:
                eval_summary = self.evaluate(episode)
                self._write_log(
                    episode=episode,
                    train_return=total_episode_reward,
                    eval_summary=eval_summary,
                )
                self.save_model(os.path.join(self.model_dir, "ppo_ep_{}.pt".format(episode)), episode=episode)
                self.logger.info(
                    "########################## PPO Policy model saved when update {} times################".format(
                        episode
                    )
                )

        self.save_model(os.path.join(self.model_dir, "ppo_final.pt"), episode=last_episode)
        self.logger.info("########################## PPO Policy model saved #################")

        self.env.close()
        self.env_test.close()

    def evaluate(self, episode):
        state = self.env_test.reset()
        terminal = False
        rewards_test = 0.0
        actions_test = []
        rule_triggers = []
        jmpcs = []
        trajectory = []
        dist_to_goal = []
        results = []

        while not terminal:
            with torch.no_grad():
                prob_a_test = self.policy_network.pi(torch.FloatTensor(state).to(self.device))
                action_test = torch.argmax(prob_a_test).item()
            state, reward_test, terminal, info = self.env_test.step(action_test)
            if self.step_info_flag:
                self._log_step(self._format_step_info(len(actions_test) + 1, info))
            rewards_test += reward_test
            actions_test.append(action_test)
            rule_triggers.append(info["rule_triggered"])
            jmpcs.append(info["jmpc"])
            trajectory.append(self.env_test.get_state(return_index=1))
            dist_to_goal.append(
                float(np.linalg.norm(np.array(trajectory[-1]) - np.array(self.env_test.goal)))
            )
            results.append(info["result"])

        action_freq = sum(actions_test) / max(len(actions_test), 1)
        rule_freq = sum(rule_triggers) / max(len(rule_triggers), 1)
        result = results[-1] if results else "Unknown"
        success = 1 if result == "Reach Goal" else 0
        self.logger.info(
            self._episode_summary(
                self.env_test,
                episode,
                len(actions_test),
                rewards_test,
                result,
                float(success),
            )
        )
        self._print_episode_result(rewards_test, result)

        self.writer.add_scalar("Eval/steps", len(actions_test), episode)
        self.writer.add_scalar("Eval/EpisodeReturns", rewards_test, episode)
        self.writer.add_scalar("Eval/action_ratio", action_freq, episode)
        self.writer.add_scalar("Eval/rule_trigger_ratio", rule_freq, episode)
        self.writer.add_scalar("Eval/success", success, episode)

        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)
        savemat(
            os.path.join(self.log_dir, "results", "ppo_results_{}.mat".format(episode)),
            {
                "act": actions_test,
                "trajectory": trajectory,
                "dist_to_goal": dist_to_goal,
                "jmpcs": jmpcs,
                "success": success,
                "reward_mode": self.env.reward_mode,
                "obs_mode": self.env.obs_mode,
            },
        )

        return {
            "return": rewards_test,
            "steps": len(actions_test),
            "success": success,
            "action_freq": action_freq,
            "rule_freq": rule_freq,
            "jmpc_sum": float(sum(jmpcs)),
        }

    def test(self, test_episodes):
        self.policy_network.eval()
        os.makedirs(os.path.join(self.log_dir, "test_results"), exist_ok=True)
        test_logger = Logger(
            os.path.join(self.log_dir, "test_episode.log"),
            clevel=logging.DEBUG,
            Flevel=logging.DEBUG,
            CMD_render=False,
            propagate=False,
        )
        statistics = Statistics(capacity=max(100, test_episodes))
        rows = []
        returns = []
        successes = []
        total_steps = 0
        total_distance = 0.0
        total_straight_distance = 0.0
        total_extra_distance_ratio = 0.0
        total_jerk_rms = 0.0
        mpc_update_count = 0
        mpc_failure_count = 0
        mpc_reuse_count = 0
        total_mpc_computations = 0
        collision_count = 0
        timeout_count = 0
        episodes_trajectory = []
        episodes_velocity = []
        episodes_acceleration = []
        episodes_trigger_seq = []
        episodes_actual_trigger_seq = []
        episodes_dist_to_goal = []
        episodes_q_weights = []
        episode_details = []

        for episode in range(1, test_episodes + 1):
            state = self.env_test.reset()
            terminal = False
            episode_return = 0.0
            actions = []
            rule_triggers = []
            jmpcs = []
            trajectory = []
            dist_to_goal = []
            task_rewards = []
            cost_rewards = []
            results = []
            actual_triggers = []
            mpc_successes = []
            q_weights = []
            velocities = []
            accelerations = []
            episode_total_distance = 0.0
            last_pos = np.array(self.env_test.get_state(return_index=1), dtype=np.float32)
            last_vel = np.zeros(3, dtype=np.float32)
            start_pos = last_pos.copy()

            while not terminal:
                with torch.no_grad():
                    prob = self.policy_network.pi(torch.FloatTensor(state).to(self.device))
                    action = torch.argmax(prob).item()
                state, reward, terminal, info = self.env_test.step(action)
                if self.step_info_flag:
                    self._log_step(self._format_step_info(len(actions) + 1, info))
                episode_return += reward
                actions.append(action)
                rule_triggers.append(info["rule_triggered"])
                actual_triggers.append(info["triggered"])
                mpc_successes.append(info["mpc_success"])
                jmpcs.append(info["jmpc"])
                task_rewards.append(info["task_reward"])
                cost_rewards.append(info["mpc_cost_reward"])
                current_pos = np.array(self.env_test.get_state(return_index=1), dtype=np.float32)
                step_distance = float(np.linalg.norm(current_pos - last_pos))
                episode_total_distance += step_distance
                current_vel = (current_pos - last_pos) / self.env_test.dt
                current_acc = (current_vel - last_vel) / self.env_test.dt
                last_pos = current_pos.copy()
                last_vel = current_vel.copy()

                trajectory.append(current_pos.tolist())
                velocities.append(current_vel.tolist())
                accelerations.append(current_acc.tolist())
                dist_to_goal.append(
                    float(np.linalg.norm(current_pos - np.array(self.env_test.goal)))
                )
                q_weights.append([float(v) for v in self.env_test.fixed_q])
                results.append(info["result"])

            success = 1 if results and results[-1] == "Reach Goal" else 0
            action_freq = sum(actions) / max(len(actions), 1)
            rule_freq = sum(rule_triggers) / max(len(rule_triggers), 1)
            actual_trigger_freq = sum(actual_triggers) / max(len(actual_triggers), 1)
            returns.append(episode_return)
            successes.append(success)
            total_steps += len(actions)
            total_distance += episode_total_distance
            mpc_updates = int(sum(actual_triggers))
            mpc_failures = int(sum(1 for v in mpc_successes if v is False))
            mpc_reuses = int(max(len(actions) - mpc_updates, 0))
            mpc_computations = int(sum(1 for v in mpc_successes if v is not None))
            mpc_update_count += mpc_updates
            mpc_failure_count += mpc_failures
            mpc_reuse_count += mpc_reuses
            total_mpc_computations += mpc_computations
            result = results[-1] if results else "Unknown"
            if "crash" in result.lower() or "collision" in result.lower():
                collision_count += 1
            if "time" in result.lower() or "timeout" in result.lower():
                timeout_count += 1

            straight_distance = max(
                float(np.linalg.norm(start_pos - np.array(self.env_test.goal, dtype=np.float32))) - 2.0,
                0.0,
            )
            if straight_distance > 0:
                total_straight_distance += straight_distance
                total_extra_distance_ratio += episode_total_distance / straight_distance

            if len(accelerations) >= 2:
                accels = np.asarray(accelerations, dtype=float)
                jerk_seq = (accels[1:] - accels[:-1]) / self.env_test.dt
                jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_seq**2, axis=1))))
            else:
                jerk_rms = 0.0
            total_jerk_rms += jerk_rms

            if "crash" in result.lower() or "collision" in result.lower():
                result_type = "collision"
            elif "time" in result.lower() or "timeout" in result.lower():
                result_type = "timeout"
            elif success:
                result_type = "success"
            else:
                result_type = "other"

            episodes_trajectory.append(trajectory if success else [])
            episodes_velocity.append(velocities if success else [])
            episodes_acceleration.append(accelerations if success else [])
            episodes_trigger_seq.append(actions if success else [])
            episodes_actual_trigger_seq.append(actual_triggers if success else [])
            episodes_dist_to_goal.append(dist_to_goal if success else [])
            episodes_q_weights.append(q_weights if success else [])

            if success:
                statistics.record_episode(
                    success=bool(success),
                    steps=len(actions),
                    distance=episode_total_distance,
                    mpc_updates=mpc_updates,
                    mpc_failures=mpc_failures,
                    mpc_reuses=mpc_reuses,
                    mpc_computations=mpc_computations,
                    result_type=result_type,
                    straight_distance=straight_distance,
                    jerk_rms=jerk_rms,
                    trajectory=trajectory,
                    velocities=velocities,
                    accelerations=accelerations,
                    trigger_seq=actions,
                    dist_to_goal=dist_to_goal,
                    q_weights=q_weights,
                )
            else:
                statistics.record_episode(
                    success=bool(success),
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

            result_path = os.path.join(
                self.log_dir, "test_results", "test_episode_{}.mat".format(episode)
            )
            savemat(
                result_path,
                {
                    "act": actions,
                    "actual_trigger": actual_triggers,
                    "trajectory": trajectory,
                    "velocities": velocities,
                    "accelerations": accelerations,
                    "dist_to_goal": dist_to_goal,
                    "jmpcs": jmpcs,
                    "task_rewards": task_rewards,
                    "mpc_cost_rewards": cost_rewards,
                    "success": success,
                    "reward_mode": self.env.reward_mode,
                    "obs_mode": self.env.obs_mode,
                },
            )

            rows.append(
                [
                    episode,
                    "{:.3f}".format(episode_return),
                    len(actions),
                    success,
                    "{:.3f}".format(action_freq),
                    "{:.3f}".format(rule_freq),
                    "{:.3f}".format(float(sum(jmpcs))),
                    result,
                ]
            )
            test_logger.info(
                self._episode_summary(
                    self.env_test,
                    episode,
                    len(actions),
                    episode_return,
                    result,
                    float(np.mean(successes)) if successes else 0.0,
                )
            )
            self._print_episode_result(episode_return, result)
            episode_details.append(
                {
                    "episode": episode,
                    "return": float(episode_return),
                    "steps": len(actions),
                    "success": bool(success),
                    "result": result,
                    "distance": float(episode_total_distance),
                    "straight_distance": float(straight_distance),
                    "extra_distance_ratio": (
                        float(episode_total_distance / straight_distance)
                        if straight_distance > 0
                        else None
                    ),
                    "jerk_rms": float(jerk_rms),
                    "mpc_updates": mpc_updates,
                    "mpc_failures": mpc_failures,
                    "mpc_reuses": mpc_reuses,
                    "mpc_computations": mpc_computations,
                    "action_freq": float(action_freq),
                    "actual_trigger_freq": float(actual_trigger_freq),
                    "rule_trigger_freq": float(rule_freq),
                    "mpc_cost_sum": float(sum(jmpcs)),
                    "trajectory": trajectory,
                    "velocities": velocities,
                    "accelerations": accelerations,
                    "trigger_seq": actions,
                    "actual_trigger_seq": actual_triggers,
                    "dist_to_goal": dist_to_goal,
                    "q_weights": q_weights,
                }
            )

        with open(os.path.join(self.log_dir, "test_logs.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Episode",
                    "Return",
                    "Steps",
                    "Success",
                    "ActionFreq",
                    "RuleTriggerFreq",
                    "MpcCostSum",
                    "Result",
                ]
            )
            writer.writerows(rows)
            writer.writerow([])
            writer.writerow(["MeanReturn", "{:.3f}".format(float(np.mean(returns)) if returns else 0.0)])
            writer.writerow(["SuccessRate", "{:.3f}".format(float(np.mean(successes)) if successes else 0.0)])

        stats = statistics.get_statistics_dict(recent=False)
        stats.update(
            {
                "reward_mode": self.env.reward_mode,
                "obs_mode": self.env.obs_mode,
                "map_index": int(self.env.map_index),
                "mean_return": float(np.mean(returns)) if returns else 0.0,
                "episodes_actual_trigger_seq": episodes_actual_trigger_seq,
                "episode_details": episode_details,
            }
        )

        json_path = os.path.join(self.log_dir, "ppo_test_statistics.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
        test_logger.info("Saved PPO test statistics json to %s" % json_path)

        self.env_test.close()
        return {
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "json_path": json_path,
        }

    def save_model(self, path, episode=None):
        torch.save(
            {
                "policy_state_dict": self.policy_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "episode": episode,
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "reward_mode": self.env.reward_mode,
                "obs_mode": self.env.obs_mode,
            },
            path,
        )

    def load_model(self, path, resume_training=False):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        self.policy_network.load_state_dict(state_dict)
        if resume_training:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.resume_episode = int(checkpoint.get("episode", 0) or 0)
            self.policy_network.train()
        else:
            self.resume_episode = 0
            self.policy_network.eval()

    def update_network(self):
        states = torch.FloatTensor(np.asarray(self.memory["state"], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(self.memory["action"]).to(self.device)
        old_probs_a = torch.FloatTensor(self.memory["action_prob"]).to(self.device)
        advantages = torch.FloatTensor(self.memory["advantage"]).to(self.device)
        td_target = self.memory["td_target"].to(self.device)

        pi = self.policy_network.pi(states)
        new_probs_a = torch.gather(pi, 1, actions)
        ratio = torch.exp(torch.log(new_probs_a + 1e-8) - torch.log(old_probs_a + 1e-8))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        pred_v = self.policy_network.v(states)
        v_loss = 0.5 * (pred_v - td_target).pow(2)
        entropy = torch.distributions.Categorical(pi).entropy().view(-1, 1)
        self.loss = (
            -torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy
        ).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, state, action, reward, next_state, terminal, prob):
        if self.memory["count"] < self.rollout_memory_size:
            self.memory["count"] += 1
        else:
            for key in [
                "state",
                "action",
                "reward",
                "next_state",
                "terminal",
                "action_prob",
                "advantage",
            ]:
                self.memory[key] = self.memory[key][1:]
            self.memory["td_target"] = self.memory["td_target"][1:]

        self.memory["state"].append(np.asarray(state, dtype=np.float32))
        self.memory["action"].append([int(action)])
        self.memory["reward"].append([float(reward)])
        self.memory["next_state"].append(np.asarray(next_state, dtype=np.float32))
        self.memory["terminal"].append([1.0 - float(terminal)])
        self.memory["action_prob"].append([float(prob)])

    def clear_memory(self):
        self.memory = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "action_prob": [],
            "terminal": [],
            "count": 0,
            "advantage": [],
            "td_target": torch.FloatTensor([]),
        }

    def finish_path(self, length):
        state = torch.FloatTensor(np.asarray(self.memory["state"][-length:], dtype=np.float32)).to(self.device)
        reward = torch.FloatTensor(self.memory["reward"][-length:]).to(self.device)
        next_state = torch.FloatTensor(
            np.asarray(self.memory["next_state"][-length:], dtype=np.float32)
        ).to(self.device)
        terminal = torch.FloatTensor(self.memory["terminal"][-length:]).to(self.device)

        td_target = reward + self.gamma * self.policy_network.v(next_state) * terminal
        delta = (td_target - self.policy_network.v(state)).detach().cpu().numpy()

        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        td_target_cpu = td_target.detach().cpu()
        if self.memory["td_target"].shape == torch.Size([1, 0]):
            self.memory["td_target"] = td_target_cpu
        else:
            self.memory["td_target"] = torch.cat((self.memory["td_target"], td_target_cpu), dim=0)
        self.memory["advantage"] += advantages

    def _write_log(self, episode, train_return, eval_summary):
        with open(os.path.join(self.log_dir, "train_logs.csv"), "a+", newline="") as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(
                [
                    "PPO-eMPC",
                    episode,
                    self.env.reward_mode,
                    self.env.obs_mode,
                    "{:.3f}".format(train_return),
                    "{:.3f}".format(eval_summary["return"]),
                    eval_summary["steps"],
                    eval_summary["success"],
                    "{:.3f}".format(eval_summary["action_freq"]),
                    "{:.3f}".format(eval_summary["rule_freq"]),
                    "{:.3f}".format(eval_summary["jmpc_sum"]),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description="PPO event-triggered MPC for AirSim drone navigation")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--reward_mode", choices=["task", "mpc_cost"], default="task")
    parser.add_argument("--obs_mode", choices=["task", "origin_like"], default="task")
    parser.add_argument("--map_index", type=int, default=102)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=200)
    parser.add_argument("--total_episodes", type=int, default=500)
    parser.add_argument("--test_episodes", type=int, default=20)
    parser.add_argument("--load_model", type=str, default=None,
                        help="Optional explicit path to a saved .pt model")
    parser.add_argument("--load_run_name", type=str, default="",
                        help="runs subdirectory name used to build save_model/ppo_*.pt")
    parser.add_argument("--load_episode", default="final",
                        help='Model episode to load, for example 335 or "final"')
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --load_model checkpoint (use with --phase train)")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trigger_steps_before_end", type=int, default=9)
    parser.add_argument("--fixed_points", action="store_true", default=True)
    parser.add_argument("--random_points", action="store_false", dest="fixed_points")
    parser.add_argument("--show_predicted_trajectory", action="store_true")
    parser.add_argument("--settle_time", type=float, default=0.2)
    parser.add_argument("--disp_info", action="store_true",
                        help="Print episode summaries in the terminal")
    parser.add_argument("--step_info", action="store_true",
                        help="Print detailed step-level action information")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if USE_CODE_CONFIG:
        args.phase = CODE_PHASE
        args.reward_mode = CODE_REWARD_MODE
        args.obs_mode = CODE_OBS_MODE
        args.map_index = CODE_MAP_INDEX
        args.index = CODE_INDEX
        args.total_episodes = CODE_TOTAL_EPISODES
        args.test_episodes = CODE_TEST_EPISODES
        args.load_run_name = CODE_LOAD_RUN_NAME
        args.load_episode = CODE_LOAD_EPISODE
        args.load_model = build_model_path(args.load_run_name, args.load_episode) if CODE_LOAD_MODEL else None
        args.resume = CODE_RESUME
        args.cuda = CODE_CUDA
        args.show_predicted_trajectory = CODE_SHOW_PREDICTED_TRAJECTORY
        args.trigger_steps_before_end = CODE_TRIGGER_STEPS_BEFORE_END
        args.fixed_points = CODE_FIXED_POINTS
        args.settle_time = CODE_SETTLE_TIME
        args.disp_info_flag = CODE_DISP_INFO_FLAG
        args.step_info_flag = CODE_STEP_INFO_FLAG
    else:
        args.disp_info_flag = args.disp_info
        args.step_info_flag = args.step_info

    rank, world_size = get_mpi_rank()
    args.rank = rank
    args.world_size = world_size
    if world_size > 1 and rank != 0:
        print(
            "[MPI] Rank %d/%d idle due to single-drone requirement. Only rank 0 will run the PPO drone environment."
            % (rank, world_size)
        )
        sys.exit(0)
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.load_model is None and args.load_run_name:
        args.load_model = build_model_path(args.load_run_name, args.load_episode)
    if args.load_model is not None:
        args.load_model = str(normalize_model_path(args.load_model))
        if not os.path.exists(args.load_model):
            raise FileNotFoundError("模型文件不存在: {}".format(args.load_model))

    dir_name, append_train_log = build_run_dir(args)
    args.append_train_log = append_train_log
    writer = SummaryWriter(log_dir=dir_name)
    agent = Agent(dir_name, writer, args, device)
    if args.phase == "test":
        if args.load_model is None:
            raise ValueError("--load_model is required when --phase test")
        agent.load_model(args.load_model)
        emit_startup_info(agent, args, dir_name, append_train_log)
        summary = agent.test(args.test_episodes)
        print(
            "Test finished. mean_return={:.3f}, success_rate={:.3f}".format(
                summary["mean_return"], summary["success_rate"]
            )
        )
    else:
        if args.load_model is not None:
            agent.load_model(args.load_model, resume_training=args.resume)
        emit_startup_info(agent, args, dir_name, append_train_log)
        agent.train(args.total_episodes)

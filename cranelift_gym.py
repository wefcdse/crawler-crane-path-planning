import math
import gymnasium as gym
from normalized_env import NormalizedEnv
from gymnasium import spaces
import numpy as np
from typing import Any, SupportsFloat
from torch.utils.tensorboard.writer import SummaryWriter

from numpy import int64
from utils import Timer

dtype = np.float32


class CraneLiftEnv(gym.Env[np.ndarray, int64]):
    def __init__(
        self,
        normalized_env: NormalizedEnv,
        max_steps: int = 400,
        tb_writer: SummaryWriter | None = None,
        log_tag: str = "Env",
    ):
        self.env = normalized_env
        self.now = self.env.random_point()
        self.target = self.env.random_point()
        self.last_position = self.now

        self.now_actionp1 = 0
        self.action_space = spaces.Discrete(6)
        self.max_steps = max_steps
        self.steps = 0
        # TensorBoard writer (optional)
        self.tb_writer = tb_writer
        self.log_tag = log_tag
        self.total_steps = 0
        #  \observation_space 是一个30个浮点数的序列
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.observe()),),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.now = self.env.random_point()
        self.target = self.env.random_point()
        self.last_position = self.now
        self.now_actionp1 = 0
        self.steps = 0
        self.best_length = 0.0
        self.inited = False
        return np.array(self.observe(), dtype=dtype), {}

    def observe(self, toc: Timer = Timer()) -> list[float]:
        with toc:
            dirs = self.env.observe_collide_directions(self.now)
        ob_collide = self.env.observe_collide(self.now, dirs)

        ob_next_step_possible = self.env.observe_next_step_possible(self.now)
        ob_last_action = [
            1.0 if i == self.now_actionp1 - 1 else 0.0 for i in range(6)
        ]  # 6
        ob_offs = self.env.pose_diff(self.now, self.target)  # 3
        ob_valid = 1.0 if self.env.valid(self.now) else 0.0  # 1
        ob_reach_goal = 1.0 if self.env.neared(self.now, self.target) else 0.0  # 1
        return [
            *ob_collide,
            *ob_next_step_possible,
            *ob_last_action,
            *ob_offs,
            ob_valid,
            ob_reach_goal,
        ]

    def observe_low(self, toc: Timer = Timer()) -> list[float]:
        with toc:
            dirs = self.env.observe_collide_directions(self.now)
        ob_collide = [0.0 for _ in self.env.observe_collide(self.now, dirs)]

        ob_next_step_possible = [
            0.0 for _ in self.env.observe_next_step_possible(self.now)
        ]
        ob_last_action = [0.0 for _ in range(6)]  # 6
        ob_offs = [-math.pi, -math.pi, -1.0]  # 3
        ob_valid = 0.0  # 1
        ob_reach_goal = 0.0  # 1
        return [
            *ob_collide,
            *ob_next_step_possible,
            *ob_last_action,
            *ob_offs,
            ob_valid,
            ob_reach_goal,
        ]

    def observe_high(self, toc: Timer = Timer()) -> list[float]:
        with toc:
            dirs = self.env.observe_collide_directions(self.now)
        ob_collide = [1.0 for _ in self.env.observe_collide(self.now, dirs)]

        ob_next_step_possible = [
            1.0 for _ in self.env.observe_next_step_possible(self.now)
        ]
        ob_last_action = [1.0 for _ in range(6)]  # 6
        ob_offs = [math.pi, math.pi, 1.0]  # 3
        ob_valid = 1.0  # 1
        ob_reach_goal = 1.0  # 1
        return [
            *ob_collide,
            *ob_next_step_possible,
            *ob_last_action,
            *ob_offs,
            ob_valid,
            ob_reach_goal,
        ]

    @staticmethod
    def observation_to_action(observation: list[float] | np.ndarray) -> int:
        """根据观察结果返回动作索引"""

        # 获取观察结果中的动作索引
        action_index = np.argmax(observation[-11:-5])
        # print(f"action_index: {observation[-11:-5]}")
        # 获取动作索引对应的动作
        return int(action_index)

    def action_to_action_index_and_step_length_signed(
        self, action: int
    ) -> tuple[int, float]:
        """0 - 2: 0 - 2(+), 3 - 5: 0 - 2(-), 0没有"""
        idx = action % 3
        direction = 1.0 if action < 3 else -1.0
        return idx, direction * self.env.step_length

    def reward(self, action: int, collide: bool, reach_goal: bool) -> float:
        out = 0.0
        # 基础消耗
        out += -self.env.step_length * 3.0
        # 发生碰撞
        if collide:
            out += -3.0
        # 到达目标
        if reach_goal:
            out += 5.0
        # 动作切换
        if self.now_actionp1 != 0 and (self.now_actionp1 - 1) != action:
            out += -self.env.step_length * 2 * 3.0
        # 靠近目标
        # last_length = np.linalg.norm(
        #     np.array(self.last_position) - np.array(self.target)
        # )
        now_length = np.linalg.norm(np.array(self.now) - np.array(self.target))

        # print(float(last_length - now_length) * 20)
        if now_length < self.best_length:
            out += float(self.best_length - now_length) * 100
            assert float(self.best_length - now_length) >= 0.0
            self.best_length = now_length
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                f"{self.log_tag}/best_length", self.best_length, self.total_steps
            )
            self.tb_writer.add_scalar(
                f"{self.log_tag}/now_length", now_length, self.total_steps
            )
            self.tb_writer.add_scalar(f"{self.log_tag}/out", out, self.total_steps)
            self.tb_writer.add_scalar(
                f"{self.log_tag}/init", self.inited, self.total_steps
            )
        if not self.inited:
            self.best_length = now_length
            self.inited = True
        return out

    def rwd_lh(self):
        return [-self.env.step_length * 3 - 10.0, 40.0 - self.env.step_length]

    def reach_goal(self) -> bool:
        return self.env.neared(self.now, self.target)

    def step(
        self, action: int, obstimer: Timer = Timer(), toc: Timer = Timer()
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        aidx, stls = self.action_to_action_index_and_step_length_signed(int(action))
        new_pos = self.env.apply_action(self.now, aidx, stls)
        collide = not self.env.valid(new_pos)
        reach_goal = self.reach_goal()
        self.now = new_pos

        reward = self.reward(int(action), collide, reach_goal)
        # TensorBoard: log reward per environment step if writer provided
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                f"{self.log_tag}/reward", reward, self.total_steps
            )
        terminated = collide or reach_goal
        truncated = self.steps >= self.max_steps

        self.now_actionp1 = action + 1
        self.steps += 1
        self.total_steps += 1
        with obstimer:
            observe = self.observe(toc=toc)

        self.last_position = self.now
        return np.array(observe, dtype=dtype), reward, terminated, truncated, {}

    def rrt(self) -> list[tuple[list[float], int, float]] | None:
        path = self.env.run_rrt(self.now, self.target)
        if path is None:
            return None
        return path

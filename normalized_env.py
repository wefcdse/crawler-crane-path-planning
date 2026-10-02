import crane_env_pyo3
import math
import numpy as np
from typing import Sequence
from utils import Timer


# 限制数字在-pi到pi之间
def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class NormalizedEnv:
    env: crane_env_pyo3.Env

    def __init__(
        self,
        env_path: str | crane_env_pyo3.Env,
        crane_position: Sequence[float],
        crane_length: float,
        crane_yaw_offs: float,
        crane_max_yaw: float = 80.0,
        crane_min_string_length: float = 0.00,
        split: int = 200,
        offs_high: float = 5.0,
        offs_low: float = 5.0,
        offs: float = 0.1,
        wider: float = 0.5,
        step_length: float = 0.01,
        # 阈值
        end_threshold: float = 0.03,
        rrt动作保持强度: float = 10,
        rrt动作切换惩罚系数: float = 50,
        rrt路径生成次数: int = 1000,
        rrt最大尝试采样次数: int = 4000,
        rrt初始动作: int = 2,
    ):
        if isinstance(env_path, crane_env_pyo3.Env):
            self.env = env_path
        else:
            self.env = crane_env_pyo3.load_env(
                path=env_path,
                split=split,
                offs_high=offs_high,
                offs_low=offs_low,
                offs=offs,
                wider=wider,
            )
        self.step_length = step_length
        self.base = crane_position
        self.size = crane_length
        self.end_threshold = end_threshold
        self.crane = crane_env_pyo3.Crane(
            crane_position,
            yaw_offs=crane_yaw_offs,
            length=crane_length,
            max_yaw=crane_max_yaw,
            min_string_length=crane_min_string_length,
        )
        self.rrt_config = crane_env_pyo3.RRTConfig(
            rrt步长=step_length * self.size,
            rrt动作保持强度=rrt动作保持强度,
            rrt终止点最大距离=end_threshold * self.size,
            rrt动作切换惩罚系数=rrt动作切换惩罚系数,
            rrt路径生成次数=rrt路径生成次数,
            rrt最大尝试采样次数=rrt最大尝试采样次数,
            rrt初始动作=rrt初始动作,
        )

    def valid(self, position: Sequence[float]) -> bool:
        """判断点是否合法，包含起重机和点的碰撞"""
        return self.crane.valid(self.env, self.to_real_position(position)) and not self.env.collide(
            self.to_real_position(position)
        )

    def neared(self, p1: Sequence[float], p2: Sequence[float]) -> bool:
        #   if ([next_p[X], next_p[Y], 0.], [end[X], end[Y], 0.]).length2()
        #         <= self.rrt终止点最大距离 * self.rrt终止点最大距离
        #         && (next_p[Z] - end[Z]).abs() < self.rrt终止点最大距离 * 2.
        #         && (!area.collide_line(next_p, end)
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (
            (x1 - x2) ** 2 + (y1 - y2) ** 2 <= self.end_threshold**2
            and (z1 - z2) ** 2 <= (self.end_threshold * 2) ** 2
            and not self.env.collide_line(self.to_real_position(p1), self.to_real_position(p2))
        )

    def random_point(self) -> list[float]:
        rand = self.env.random_point()
        rp = self.from_real_position(rand)
        if (
            not self.valid(rp)
            or not self.valid(self.apply_action(rp, 0, -self.step_length))
            or not self.valid(self.apply_action(rp, 0, self.step_length))
            or not self.valid(self.apply_action(rp, 1, -self.step_length))
            or not self.valid(self.apply_action(rp, 1, self.step_length))
            or not self.valid(self.apply_action(rp, 2, -self.step_length))
            or not self.valid(self.apply_action(rp, 2, self.step_length))
        ):
            return self.random_point()
        return rp

    def normals(self, position: Sequence[float]):
        real_position = self.to_real_position(position)
        n0, n1, n2 = self.crane.normals(real_position)
        n1[2] = 0.0
        n1l = math.sqrt(n1[0] ** 2 + n1[1] ** 2)
        n1[0] /= n1l
        n1[1] /= n1l
        return n0, n1, n2

    def to_real_position(self, position: Sequence[float]) -> list[float]:
        return [
            position[0] * self.size + self.base[0],
            position[1] * self.size + self.base[1],
            position[2] * self.size + self.base[2],
        ]

    def from_real_position(self, real_position: Sequence[float]) -> list[float]:
        return [
            (real_position[0] - self.base[0]) / self.size,
            (real_position[1] - self.base[1]) / self.size,
            (real_position[2] - self.base[2]) / self.size,
        ]

    def pose_diff(self, start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        pass
        real_start = self.to_real_position(start)
        real_end = self.to_real_position(end)
        pose_start = self.crane.position_to_pose(real_start)
        pose_end = self.crane.position_to_pose(real_end)
        return [
            normalize_angle(pose_end[0] - pose_start[0]),
            normalize_angle(pose_end[1] - pose_start[1]),
            (pose_end[2] - pose_start[2]) / self.size,
        ]

    def observe_collide_directions1(self, position: Sequence[float], count: int = 8) -> list[list[float]]:
        """观察的方向的列表"""
        pass
        n0, n1, n2 = self.normals(position)
        n0, n1, n2 = np.array(n0), np.array(n1), np.array(n2)
        out = []

        def normalize(x: np.ndarray) -> np.ndarray:
            return x / np.linalg.norm(x)

        # 八个角
        out.append(normalize(n0 + n1 + n2))
        out.append(normalize(n0 + n1 - n2))
        out.append(normalize(n0 - n1 + n2))
        out.append(normalize(n0 - n1 - n2))
        out.append(normalize(-n0 + n1 + n2))
        out.append(normalize(-n0 + n1 - n2))
        out.append(normalize(-n0 - n1 + n2))
        out.append(normalize(-n0 - n1 - n2))

        # 12条棱边
        def appendline(neg0: bool, neg1: bool, neg2: bool):
            for i in range(count):
                fix0 = -1 if neg0 else 1
                fix1 = -1 if neg1 else 1
                fix2 = -1 if neg2 else 1
                out.append(normalize((n0 * fix0 + n1 * fix1 + n2 * fix2) * (count + 1) / 2 - n0 * (i + 1) * fix0))
                out.append(normalize((n0 * fix0 + n1 * fix1 + n2 * fix2) * (count + 1) / 2 - n1 * (i + 1) * fix1))
                out.append(normalize((n0 * fix0 + n1 * fix1 + n2 * fix2) * (count + 1) / 2 - n2 * (i + 1) * fix2))

        appendline(False, False, False)
        appendline(True, True, False)
        appendline(True, False, True)
        appendline(False, True, True)

        # 6个面
        def appendface(base: np.ndarray, sur1: np.ndarray, sur2: np.ndarray):
            for x in range(count):
                for y in range(count):
                    out.append(
                        normalize(
                            base * (count + 1) / 2
                            - sur1 * (count + 1) / 2
                            - sur2 * (count + 1) / 2
                            + sur1 * (x + 1)
                            + sur2 * (y + 1)
                        )
                    )

        appendface(n0, n1, n2)
        appendface(-n0, n1, n2)
        appendface(n1, n0, n2)
        appendface(-n1, n0, n2)
        appendface(n2, n0, n1)
        appendface(-n2, n0, n1)
        maped = np.array(out).tolist()

        return maped

    def observe_collide_directions(self, position: Sequence[float], count: int = 8) -> list[list[float]]:
        """观察的方向的列表"""
        pass
        n0, n1, n2 = self.normals(position)
        out = crane_env_pyo3.dim_to_sample(n0, n1, n2, count)
        return out

    def observe_collide(
        self,
        position: Sequence[float],
        directions: Sequence[Sequence[float]],
        dir_timer=Timer(),
        collide_timer=Timer(),
    ) -> Sequence[float]:
        with dir_timer:
            real_position = self.to_real_position(position)
        # out = [
        #     self.env.direction_length(
        #         real_position, direction=[*direction], max=self.size
        #     )
        #     / self.size
        #     for direction in directions
        # ]
        with collide_timer:
            out = self.env.direction_length_multi_dir(real_position, directions, max=self.size)
            out = [x / self.size for x in out]
        return out

    def observe_action(self, position: Sequence[float]):
        pass
        real_position = self.to_real_position(position)
        steps: list[float] = [0, 0, 0, 0, 0, 0]
        yl = self.yaw_length(position)
        l = self.size
        steps[0] = self.crane.max_step(self.env, real_position, 0, -1) / yl
        steps[1] = self.crane.max_step(self.env, real_position, 0, 1) / yl
        steps[2] = self.crane.max_step(self.env, real_position, 1, -1) / l
        steps[3] = self.crane.max_step(self.env, real_position, 1, 1) / l
        steps[4] = self.crane.max_step(self.env, real_position, 2, -1) / l
        steps[5] = self.crane.max_step(self.env, real_position, 2, 1) / l
        return steps

    def yaw_length(self, position: Sequence[float]) -> float:
        return self.crane.yaw_length(self.to_real_position(position))

    def observe_next_step_possible(
        self, position: Sequence[float], step_length: float | None = None
    ) -> Sequence[float]:
        if step_length is None:
            step_length = self.step_length
        real_position = self.to_real_position(position)
        real_step_length = step_length * self.size
        out: list[bool] = []
        for i in range(3):
            pos = self.crane.apply_step(real_position, i, real_step_length)
            neg = self.crane.apply_step(real_position, i, -real_step_length)
            out.append(self.crane.valid(self.env, pos) and not self.env.collide_line(real_position, pos))
            out.append(self.crane.valid(self.env, neg) and not self.env.collide_line(real_position, neg))
        out1: list[float] = [1 if x else 0 for x in out]
        return out1

    def apply_action(
        self,
        position: Sequence[float],
        action_index: int,
        step_length_and_direction: float,
    ) -> Sequence[float]:
        real_position = self.to_real_position(position)
        real_step_length = step_length_and_direction * self.size
        new_position = self.crane.apply_step(real_position, action_index, real_step_length)
        return self.from_real_position(new_position)

    def run_rrt(self, start: Sequence[float], goal: Sequence[float]) -> None | list[tuple[list[float], int, float]]:
        real_start = self.to_real_position(start)
        read_goal = self.to_real_position(goal)
        path = self.rrt_config.rrt(self.crane, self.env, real_start, read_goal)
        if path is None:
            return None
        out = [(self.from_real_position(pos), action, steplength / self.size) for pos, action, steplength in path]
        return out


# 导出类
__all__ = ["NormalizedEnv"]

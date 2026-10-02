from typing import List, Optional, Sequence, Tuple
import random
from dataclasses import dataclass

from crane_env_pyo3 import Crane, Env

Vec3 = List[float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return [a[i] + b[i] for i in range(3)]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return [a[i] - b[i] for i in range(3)]


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a: Vec3) -> float:
    return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5


def length2(a: Vec3) -> float:
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]


def scale(a: Vec3, s: float) -> Vec3:
    return [a[i] * s for i in range(3)]


def normalize(a: Vec3) -> Vec3:
    l = length(a)
    if l == 0:
        return [0.0, 0.0, 0.0]
    return [a[i] / l for i in range(3)]


@dataclass
class _Node:
    pos: Vec3
    root: int
    from_move_idx: int
    from_move_step: float


@dataclass
class RRTConfig:
    rrt_step_length: float
    rrt_keep_action_strength: float
    rrt_goal_dist: float
    rrt_max_sample_attempts: int
    rrt_initial_action: int

    def rrt_once(
        self,
        crane: Crane,
        env: Env,
        start: Sequence[float],
        end: Sequence[float],
    ) -> Optional[List[Tuple[Vec3, int, float]]]:
        """
        返回: 可选的路径，路径为 List[(pos, from_move_idx, from_move_step)]，第一个元素为起点或接近起点，最后一项为目标位置。
        参数类型:
        - crane: crane_env_pyo3.Crane
        - env: crane_env_pyo3.Env
        - start, end: 长度为3的序列
        """
        step_length = self.rrt_step_length
        if step_length <= 0.0:
            raise ValueError("rrt_step_length must be > 0")

        # APIs 由 crane_env_pyo3 提供（签名由绑定自动生成并保证存在）

        route: List[_Node] = [_Node(pos=list(start), root=0, from_move_idx=self.rrt_initial_action, from_move_step=0.0)]

        counter = 0
        max_count = self.rrt_max_sample_attempts

        while True:
            # 采样随机点
            p = env.random_point()

            # 最近节点选择，带有朝向目标的方向性偏置
            def nearest_idx(p_sample: Vec3) -> int:
                nearest = 0
                best = float("inf")
                for idx, node in enumerate(route):
                    p1 = node.pos
                    # 方向修正：归一化向量的点积
                    dir_to_goal = normalize([end[i] - p1[i] for i in range(3)])
                    dir_to_sample = normalize([p_sample[i] - p1[i] for i in range(3)])
                    direction_fix = dot(dir_to_goal, dir_to_sample)
                    l = length([p1[0] - p_sample[0], p1[1] - p_sample[1], p1[2] - p_sample[2]]) * (2.03 - direction_fix)
                    if l < best:
                        best = l
                        nearest = idx
                return nearest

            ni = nearest_idx(p)
            base_point = route[ni].pos
            base_idx = ni
            from_move_idx = route[ni].from_move_idx

            # 从 Crane 获取动作法向 (normals)
            (t0, t1, l) = crane.normals(base_point)
            moves = [t0, t1, l]

            # 期望方向
            direction = normalize([p[i] - base_point[i] for i in range(3)])

            # 选择最接近的动作索引
            best_score = float("-inf")
            best_idx = 0
            for idx, normal in enumerate(moves):
                d = abs(dot(normalize(normal), direction))
                bonus = 0.0
                if idx == from_move_idx:
                    bonus = self.rrt_keep_action_strength * random.random()
                else:
                    # 一个动作方向接近目标点后，选择其他动作方向时也给予一定奖励
                    if crane.neared(base_point, end, from_move_idx, step_length / 2.0):
                        bonus = self.rrt_keep_action_strength * random.random()
                score = d + bonus

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # 确定步长（带符号）
            step_sign = 1.0 if dot(moves[best_idx], direction) >= 0 else -1.0
            step = step_sign * step_length

            # 使用 Crane.apply_step 计算下一个点
            next_p = crane.apply_step(base_point, best_idx, step)

            # 使用提供的 API 进行合法性校验（异常让其自然抛出，便于调试）
            if not crane.valid(env, next_p) or env.collide(next_p) or env.collide_line(base_point, next_p):
                if not crane.mercy(env, base_point, best_idx):
                    continue

            route.append(_Node(pos=list(next_p), root=base_idx, from_move_idx=best_idx, from_move_step=step))
            counter += 1

            # 检查是否接近目标（XY 平面距离与 Z 方向容差）
            dx = route[-1].pos[0] - end[0]
            dy = route[-1].pos[1] - end[1]
            dz = route[-1].pos[2] - end[2]
            if (dx * dx + dy * dy) <= (self.rrt_goal_dist * self.rrt_goal_dist) and abs(dz) < self.rrt_goal_dist * 2.0:
                # ensure not blocked
                blocked = env.collide_line(route[-1].pos, list(end)) and not crane.mercy(env, route[-1].pos, best_idx)
                if not blocked:
                    break

            if counter > max_count:
                return None
        # build path: start with end then backtrack
        path: List[Tuple[Vec3, int, float]] = []
        # add end as first element
        path.append((list(end), 0, 0.0))

        now_idx = len(route) - 1
        while True:
            node = route[now_idx]
            path.append((list(node.pos), node.from_move_idx, node.from_move_step))
            if node.root == now_idx:
                break
            if node.root == 0 and now_idx == 0:
                break
            # protect against malformed roots
            if node.root >= len(route) or node.root < 0:
                break
            if node.root == now_idx:
                break
            now_idx = node.root

        path.reverse()
        return path

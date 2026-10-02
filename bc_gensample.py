import pickle
import torch
import numpy as np
from typing import TypeVar
from normalized_env import NormalizedEnv
from cranelift_gym import CraneLiftEnv

T = TypeVar("T")


def not_none(x):
    # type: (T | None) -> T
    if x is None:
        raise ValueError("value cannot be None")
    return x


# 训练函数
def main(gen_env_count=0, target_file="paths_complex.pkl"):

    # 创建环境
    ev = NormalizedEnv(
        "envs/测试场景.obj", [0, 0, 2], 10, 0.2, crane_min_string_length=6
    )
    env = CraneLiftEnv(ev)

    # 收集经验
    paths = []
    # 读取json
    try:
        pkl_file = open(target_file, "rb")
        paths = pickle.load(pkl_file)
    except Exception as e:
        print("error:", e)
        pass
    print("paths:", len(paths))

    for i in range(gen_env_count):
        obs, lbl = gen_samples(env)
        observations = torch.tensor(np.array(obs), dtype=torch.float32)
        labels = torch.tensor(lbl, dtype=torch.long)

        paths.append((observations.tolist(), labels.tolist()))
        print("path:", i, "len:", len(observations))
    pkl_file = open(target_file, "wb")
    pickle.dump(paths, pkl_file)
    pkl_file.close()


def gen_samples(env: CraneLiftEnv) -> tuple[list[np.ndarray], list[int]]:
    state, _ = env.reset()
    path: list[tuple[list[float], int, float]] = env.rrt()  # type: ignore
    if path is None:
        return gen_samples(env)

    states = []
    labels = []
    for _, action_idx, direction in path[1:-1]:
        action = action_idx if direction > 0 else action_idx + 3
        states.append(state)
        labels.append(action)
        state, _, _, _, _ = env.step(action)
        # print(reward)
    return states, labels


if __name__ == "__main__":
    main(gen_env_count=20)

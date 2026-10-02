import torch
import pickle
from typing import TypeVar
from cranelift_gym import CraneLiftEnv

T = TypeVar("T")


def not_none(x):
    # type: (T | None) -> T
    if x is None:
        raise ValueError("value cannot be None")
    return x


from torch.utils.data import Dataset


class Ds1(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.tensor(d) for d in data]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def load_data(path="ppo_paths") -> tuple[Ds1, Ds1]:
    full_path = path + ".pkl"
    noturn_path = path + "_noturn.pkl"
    turn_path = path + "_turn.pkl"

    try:
        pkl_file = open(noturn_path, "rb")
        obs_noturn, lab_noturn = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open(turn_path, "rb")
        obs_turn, lab_turn = pickle.load(pkl_file)
        pkl_file.close()
        print("fastload noturn:", len(obs_noturn), "turn:", len(obs_turn))

        return Ds1(obs_noturn, lab_noturn), Ds1(obs_turn, lab_turn)
    except Exception as e:
        print("error:", e)
        print("loading from original data")
        pass

    paths: list[tuple[list[list[float]], list[int]]] = []
    # 读取json
    try:
        pkl_file = open(full_path, "rb")
        paths = pickle.load(pkl_file)
        pkl_file.close()
    except Exception as e:
        print("error:", e)
        pass
    print("paths:", len(paths))
    obs_noturn = []
    lab_noturn = []
    obs_turn = []
    lab_turn = []
    for observations, labels in paths:
        for obs, lab in zip(observations, labels):
            act = CraneLiftEnv.observation_to_action(obs)
            if act == lab:
                obs_noturn.append(obs)
                lab_noturn.append(lab)
            else:
                obs_turn.append(obs)
                lab_turn.append(lab)
    pkl_file = open(noturn_path, "wb")
    pkl_file.write(pickle.dumps((obs_noturn, lab_noturn)))
    pkl_file.close()
    pkl_file = open(turn_path, "wb")
    pkl_file.write(pickle.dumps((obs_turn, lab_turn)))
    pkl_file.close()

    print("noturn:", len(obs_noturn), "turn:", len(obs_turn))
    return Ds1(obs_noturn, lab_noturn), Ds1(obs_turn, lab_turn)

import os
import pickle
import random

# 可在脚本头部修改：数据集基名 (不带扩展名)，例如 "ppo_paths" 或 "paths_complex"
DATASET_BASE = "paths_complex"
# 测试集比例
SPLIT_RATIO = 0.2
# 随机种子
RANDOM_STATE = 42
# 输出目录
OUT_DIR = "dataset"

os.makedirs(OUT_DIR, exist_ok=True)

in_file = DATASET_BASE + ".pkl"
if not os.path.isfile(in_file):
    print(f"找不到文件: {in_file}")
else:
    try:
        with open(in_file, "rb") as f:
            paths = pickle.load(f)
    except Exception as e:
        print(f"读取 {in_file} 失败: {e}")
        paths = []

    if not isinstance(paths, list) or len(paths) == 0:
        print(f"数据格式异常或为空: {in_file}")
    else:
        rng = random.Random(RANDOM_STATE)
        indices = list(range(len(paths)))
        rng.shuffle(indices)

        split_at = int(len(indices) * (1.0 - float(SPLIT_RATIO)))
        train_idx = indices[:split_at]
        test_idx = indices[split_at:]

        train = [paths[i] for i in train_idx]
        test = [paths[i] for i in test_idx]

        out_train = os.path.join(OUT_DIR, f"{DATASET_BASE}_train.pkl")
        out_test = os.path.join(OUT_DIR, f"{DATASET_BASE}_test.pkl")

        try:
            with open(out_train, "wb") as f:
                pickle.dump(train, f)
            with open(out_test, "wb") as f:
                pickle.dump(test, f)
        except Exception as e:
            print(f"保存失败: {e}")
        else:
            print(
                f"已保存: {out_train} ({len(train)} episodes), {out_test} ({len(test)} episodes)"
            )

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from normalized_env import NormalizedEnv
from cranelift_gym import CraneLiftEnv


from bc_load_data import load_data
from utils import Timer
from torch.utils.tensorboard import SummaryWriter  # type: ignore


def tensors_from_dataset(ds, batch_collect=1024):
    """Concatenate all samples from a Dataset into two tensors (obs, labels).

    This is memory-intensive for large datasets; it's fine for typical sample sizes.
    """
    obs_chunks = []
    lab_chunks = []
    loader = DataLoader(ds, batch_size=batch_collect, shuffle=False)
    for o, l in loader:
        obs_chunks.append(o)
        lab_chunks.append(l)
    if len(obs_chunks) == 0:
        return torch.empty((0,)), torch.empty((0,))
    return torch.cat(obs_chunks, dim=0), torch.cat(lab_chunks, dim=0)


def sample2(obs_a, lab_a, obs_b, lab_b, n_a: int, n_b: int, device=None):
    """从两个已 materialize 的 tensor 数据集中分别采样并拼接。

    obs_a, lab_a, obs_b, lab_b 都是 tensor；n_a/n_b 指定各自采样数量。
    返回 (batch_obs, batch_labels)，已随机打乱并移动到 `device`。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    na = obs_a.size(0)
    nb = obs_b.size(0)
    if na == 0 or nb == 0:
        raise RuntimeError("之一数据集为空，无法采样")

    # make indices live on the same device as the source tensors to avoid
    # `indices should be either on cpu or on the same device as the indexed tensor` errors
    idx_a = torch.randint(0, na, (n_a,), device=obs_a.device)
    idx_b = torch.randint(0, nb, (n_b,), device=obs_b.device)

    a_obs = obs_a[idx_a].to(device).float()
    b_obs = obs_b[idx_b].to(device).float()
    a_lab = lab_a[idx_a].to(device).long()
    b_lab = lab_b[idx_b].to(device).long()

    batch_obs = torch.cat([a_obs, b_obs], dim=0)
    batch_labels = torch.cat([a_lab, b_lab], dim=0)

    perm = torch.randperm(batch_obs.size(0), device=device)
    return batch_obs[perm], batch_labels[perm]


class NLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=[4096]):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim[0])

        self.output = torch.nn.Linear(hidden_dim[0], 6)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.input(x))

        return self.softmax(self.output(x))

    def benchmark_forward(self, count: int = 10000):
        start_time = time.time()
        with torch.no_grad():
            for _ in range(count):
                sample_input = torch.randn(1, self.input.in_features, device=next(self.parameters()).device)
                _ = self.forward(sample_input)
        end_time = time.time()
        print(f"Benchmark forward time for 100 runs: {end_time - start_time:.6f} seconds")

    def benchmark_backward(self, count: int = 1000):
        sample_input = torch.randn(1, self.input.in_features, device=next(self.parameters()).device)
        sample_output = self.forward(sample_input)
        sample_target = torch.randint(0, sample_output.size(1), (1,), device=next(self.parameters()).device)

        start_time = time.time()
        for _ in range(count):
            self.zero_grad()
            output = self.forward(sample_input)
            loss = F.cross_entropy(output, sample_target)
            loss.backward()
        end_time = time.time()
        print(f"Benchmark backward time for {count} runs: {end_time - start_time:.6f} seconds")


def eval_env_once(
    env: CraneLiftEnv,
    model: NLP,
):
    # 3. 运行一个 episode，累加奖励
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action_probs = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device))
        action: int = torch.argmax(action_probs, dim=1).item()  # type: ignore
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
        episode_reward += reward
        done = terminated or truncated
    success = env.reach_goal()
    print(f"本次运行的总奖励: {episode_reward}, 成功: {success}")
    return success


def bc_train(
    model: NLP,
    dsnt,
    dst,
    save_path: str,
    epochs=10,
    batch_per_side=64,
    lr=1e-4,
    device=None,
    tb_writer=None,
    test_dsnt=None,
    test_dst=None,
    test_sample_size=1024,
):
    """Behavior cloning pretrain using two datasets (noturn, turn).

    Each training step randomly samples `batch_per_side` items from each dataset,
    concatenates them, shuffles, and makes one optimization step with
    CrossEntropyLoss on the policy logits (discrete actions assumed).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model
    model.to(device)
    model.benchmark_forward()
    # materialize datasets as tensors for fast random indexing
    obs_nt, lab_nt = tensors_from_dataset(dsnt)
    obs_t, lab_t = tensors_from_dataset(dst)

    # materialize test datasets if provided
    if test_dsnt is not None and test_dst is not None:
        obs_nt_test, lab_nt_test = tensors_from_dataset(test_dsnt)
        obs_t_test, lab_t_test = tensors_from_dataset(test_dst)
    else:
        obs_nt_test = lab_nt_test = obs_t_test = lab_t_test = None

    n_nt = obs_nt.size(0)
    n_t = obs_t.size(0)
    if n_nt == 0 or n_t == 0:
        raise RuntimeError("Both noturn and turn datasets must be non-empty")

    batch_each = int(batch_per_side)
    steps_per_epoch = max(1, int(np.ceil(max(n_nt, n_t) / batch_each)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    global_step = 0
    for ep in range(epochs):
        with Timer(f"BC pretrain epoch {ep+1}/{epochs}", True):
            losses = []
            for _ in range(steps_per_epoch):
                batch_obs, batch_labels = sample2(obs_nt, lab_nt, obs_t, lab_t, batch_each, batch_each, device=device)

                # shuffle merged batch
                perm = torch.randperm(batch_obs.size(0), device=device)
                batch_obs = batch_obs[perm]
                batch_labels = batch_labels[perm]

                optimizer.zero_grad()

                logits = model(batch_obs)

                loss = F.cross_entropy(logits, batch_labels.squeeze())
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # Log to tensorboard every step
                if tb_writer is not None:
                    tb_writer.add_scalar("BC/loss_step", loss.item(), global_step)
                    tb_writer.add_scalar("BC/learning_rate", lr, global_step)

                global_step += 1

            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

            # Evaluate on provided test datasets (separately for noturn and turn)
            acc_noturn = None
            acc_turn = None
            if obs_nt_test is not None and obs_t_test is not None:
                # helper to sample from a single tensor dataset
                def sample_single(obs, lab, n):
                    n = min(n, obs.size(0))
                    if n == 0:
                        return None, None
                    idx = torch.randint(0, obs.size(0), (n,), device=obs.device)
                    return obs[idx].to(device).float(), lab[idx].to(device).long()

                with torch.no_grad():
                    s_obs_nt, s_lab_nt = sample_single(obs_nt_test, lab_nt_test, test_sample_size)
                    s_obs_t, s_lab_t = sample_single(obs_t_test, lab_t_test, test_sample_size)

                    if s_obs_nt is not None:
                        pred_nt = torch.argmax(model(s_obs_nt), dim=1)
                        acc_noturn = (pred_nt == s_lab_nt.squeeze()).float().mean().item()  # type: ignore
                    if s_obs_t is not None:
                        pred_t = torch.argmax(model(s_obs_t), dim=1)
                        acc_turn = (pred_t == s_lab_t.squeeze()).float().mean().item()  # type: ignore

            print_msg = f"BC epoch {ep+1}/{epochs}  loss={avg_loss:.6f}"
            if acc_noturn is not None:
                print_msg += f"  acc_noturn={acc_noturn:.4f}"
            if acc_turn is not None:
                print_msg += f"  acc_turn={acc_turn:.4f}"
            print(print_msg)

            # Log epoch-level metrics to tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar("BC/loss_epoch", avg_loss, ep)
                if acc_noturn is not None:
                    tb_writer.add_scalar("BC/accuracy_epoch_noturn", acc_noturn, ep)
                if acc_turn is not None:
                    tb_writer.add_scalar("BC/accuracy_epoch_turn", acc_turn, ep)
        if ep % 16 == 0 or ep == epochs - 1:  # Save model checkpoint every 16 epochs
            model_path = os.path.join(save_path, f"bc_model_epoch_{ep+1}.pth")
            torch.save(model.state_dict(), model_path)
    model.eval()


def main(
    epochs: int,
    batch_per_side: int,
    width: int,
    lr: float,
    name: str,
    show_name=False,
    name_map=lambda n, *_,: n,
):
    name = name_map(name, epochs, batch_per_side, width, lr)
    if show_name:
        print(f"Training model: {name}")
        return
    save_path = f"models6/{name}"
    os.makedirs(save_path, exist_ok=True)
    # copy train script to model dir for reference
    copy = lambda name: os.system(f'cp {name} "{save_path}/{name}"')
    copy("bc_only.py")
    copy("bc_load_data.py")
    copy("cranelift_gym.py")

    # Environment
    ev = NormalizedEnv("envs/测试场景_复杂.obj", [0, 0, 2], 10, 0.2, crane_min_string_length=6)
    # TensorBoard: create writer for environment logging
    env = CraneLiftEnv(ev, max_steps=1000, tb_writer=None, log_tag="Env")

    # 使用默认的MlpPolicy
    model = NLP(env.reset()[0].shape[0], hidden_dim=[width])
    model.benchmark_forward()
    model.benchmark_backward()
    model = model.to("cuda")
    model.benchmark_forward()
    model.benchmark_backward()
    # Load sample datasets (must exist)
    dsnt, dst = load_data("dataset/paths_complex_train")
    test_dsnt, test_dst = load_data("dataset/paths_complex_test")
    print("Loaded sample datasets: noturn", len(dsnt), "turn", len(dst))
    # BC pretrain with tensorboard logging
    bc_writer = SummaryWriter(log_dir=f"{save_path}/tblog/BC")
    bc_train(
        model,
        dsnt,
        dst,
        save_path=save_path,
        epochs=epochs,
        batch_per_side=batch_per_side,
        lr=lr,
        tb_writer=bc_writer,
        test_dsnt=test_dsnt,
        test_dst=test_dst,
        test_sample_size=256,
    )

    # 保存模型
    model_path = f"{save_path}/bc_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved BC model to {model_path}")

    success_count = 0
    # Close
    eval_count = 50
    for _ in range(50):
        success = eval_env_once(env, model)
        if success:
            success_count += 1
    print(f"Success count: {success_count}/{eval_count}")
    bc_writer.add_scalar("Eval/success_count", success_count, 0)
    bc_writer.add_scalar("Eval/success_rate", success_count / eval_count, 0)
    bc_writer.close()


def use_counter(start):
    counter = start

    def inner():
        nonlocal counter
        val = counter
        counter += 1
        # 输出成3位数格式
        return f"{val:03d}"

    return inner


if __name__ == "__main__":
    idx = use_counter(200)
    base_name = ""
    name_map = lambda n, e, b, w, lr: f"lr{lr*1e5:05.1f}_{base_name}_{n}_w{w}"

    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=0.5e-5, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=0.1e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=0.3e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=0.6e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=1.2e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=8192, lr=1.4e-4, name_map=name_map)

    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=0.5e-5, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=0.1e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=0.3e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=0.6e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=1.2e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=4096, lr=1.4e-4, name_map=name_map)

    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=0.5e-5, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=0.1e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=0.3e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=0.6e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=1.2e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=1024, lr=1.4e-4, name_map=name_map)

    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=0.5e-5, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=0.1e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=0.3e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=0.6e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=1.2e-4, name_map=name_map)
    main(name=f"{idx()}", epochs=256, batch_per_side=64, width=512, lr=1.4e-4, name_map=name_map)

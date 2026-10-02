# Crawler Crane Path Planning (RRT-BC)

Hybrid path planning for automated crawler-crane lifting:
a lightweight 2.5D grid height field, an **Improved RRT** with actuator
constraints embedded in node expansion, and a **Behavioral Cloning (BC)** prior
fused by a probability threshold.

## Layout

| File | Role |
|---|---|
| `rrt.py` | Improved RRT (discrete actions + geometric heuristics) |
| `rrt_bc.py` | RRT-BC (BC prior + geometric exploration) |
| `bc_only.py` | BC training (hierarchical balanced sampling) |
| `bc_gensample.py`, `bc_load_data.py`, `split_dataset.py` | Expert data gen / load / split |
| `cranelift_gym.py`, `normalized_env.py`, `custom_models.py` | Gym env, 2.5D normalized env, PPO nets |
| `src/` (`crane_env_pyo3`), `box_gen/` | Rust acceleration: height field, collision, UTM, crane kinematics |
| `models/bc_model.pth` | Trained BC model |
| `*.ipynb` | Debug / figure reproduction |

## Build

Rust module is a PyO3 extension:

```bash
maturin develop --release     # or: cargo build --release
```

Python deps: `torch numpy gymnasium stable_baselines3 matplotlib tensorboard`

## Run

```bash
python bc_gensample.py     # collect expert trajectories
python bc_only.py          # train BC model
# see *.ipynb for RRT-BC evaluation
```

## Notes

- Scene `*.obj` and datasets (`dataset/*.pkl`) are **not included**; place your own under `envs/` and `dataset/`.
- Coordinates in `box_gen/config_default.json` are placeholders.

## License

TBD.

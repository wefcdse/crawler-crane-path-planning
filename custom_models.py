"""Custom policy and network architectures for PPO training.

Provides a unified custom model with fixed structure. Modify the network
definitions directly in CustomExtractor and CustomActorCritic classes.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3 import PPO
import gymnasium as gym


class CustomExtractor(BaseFeaturesExtractor):
    """特征提取器 - 将原始观测转换为特征向量

    作用：
        - 接收环境的原始观测（如传感器数据）
        - 通过多层神经网络进行处理
        - 输出 features_dim 维的特征向量供 Actor-Critic 使用

    如何修改：
        - 改变隐层大小：修改 nn.Linear() 的参数
        - 改变层数：增删 nn.Linear() 和 nn.ReLU() 层
        - 示例：改为更深的网络
          nn.Linear(input_size, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU(),
          ...以此类推
    """

    def __init__(self, observation_space: spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        shape = observation_space.shape
        input_size = int(np.prod(shape)) if shape is not None else 0

        # Customize network architecture here
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CustomActorCritic(nn.Module):
    """Actor-Critic 网络 - 策略和价值评估的统一网络

    作用：
        - 共享特征处理 (self.body)：所有特征都通过相同的中间层
        - 分离 Actor 头 (policy_net)：转换为中间表示 latent_pi
        - 分离 Critic 头 (value_net)：转换为中间表示 latent_vf

    如何修改：
        - 改变共享体大小：修改 self.body 的 nn.Linear() 参数
        - 改变中间层维度：修改 latent_dim 参数或改 policy_net/value_net
        - 示例：变为更大的网络
          self.body = nn.Sequential(
              nn.Linear(features_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 128),
              nn.ReLU(),
          )
    """

    def __init__(self, features_dim: int, action_dim: int, latent_dim: int = 128):
        super().__init__()

        self.latent_dim = latent_dim

        # Customize shared body here
        self.body = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent_pi, latent_vf) - compatible with SB3 standard"""
        x = self.body(features)
        # Return same latent representation for both policy and value
        # Final heads (action_net, value_net) are in CustomPolicy
        return x, x


class CustomPolicy(ActorCriticPolicy):
    """自定义策略类 - 整合特征提取和 Actor-Critic 的主类

    作用：
        - 创建并管理特征提取器 (CustomExtractor)
        - 创建并管理 Actor-Critic 网络 (CustomActorCritic)
        - 实现前向传播、动作采样、价值预测等核心方法
        - 与 Stable-Baselines3 的 PPO 算法兼容

    通常无需修改此类，除非需要调整：
        - 特征提取器的参数（features_dim 大小）
        - 动作空间的处理方式
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        features_dim: int = 256,
        latent_dim: int = 128,
        **kwargs,
    ):
        self.custom_features_dim = features_dim
        self.latent_dim = int(latent_dim)

        # Don't call super().__init__() yet - we need to configure features_extractor_class first
        # We'll manually set up what we need before the parent init
        self.features_dim_override = features_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        # Replace the default features_extractor with our custom one
        self.features_extractor = CustomExtractor(
            observation_space, features_dim=features_dim
        )

        # Now update features_dim to match
        self._features_dim = features_dim

        # Get action dimension
        if isinstance(self.action_space, spaces.Discrete):
            action_dim = int(self.action_space.n)
        else:
            shape = self.action_space.shape
            action_dim = int(np.prod(shape)) if shape is not None else 0

        # Create actor-critic network
        # latent_dim: 中间层维度，可在 policy_kwargs 或 create_ppo_model() 中配置
        self.mlp_extractor = CustomActorCritic(
            features_dim, action_dim, latent_dim=self.latent_dim
        )

        # Setup action distribution - action_net processes latent_pi to logits
        self.action_net = nn.Linear(self.latent_dim, action_dim)
        self.action_dist = CategoricalDistribution(action_dim)

        # Setup value function - value_net processes latent_vf to scalar
        self.value_net = nn.Linear(self.latent_dim, 1)

        # Build networks
        self.to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns (actions, values, log_probs)"""
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Process through final heads
        policy_logits = self.action_net(latent_pi)
        values = self.value_net(latent_vf)

        distribution = self.action_dist.proba_distribution(policy_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution"""
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        policy_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(policy_logits)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values"""
        features = self.features_extractor(obs)
        _, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        return values


def create_ppo_model(
    env: gym.Env,
    learning_rate: float = 3e-4,
    device: str = "auto",
    verbose: int = 0,
    features_dim: int = 256,
    latent_dim: int = 128,
    **ppo_kwargs,
) -> PPO:
    """创建 PPO 模型的工厂函数

    作用：
        - 简化模型创建过程
        - 自动配置 CustomPolicy、CustomExtractor、CustomActorCritic
        - 集成 PPO 训练算法

    参数说明：
        - env: 训练环境
        - learning_rate: 学习率（默认 3e-4）
        - device: 设备选择 ('cpu'、'cuda'、'auto')
        - verbose: 日志级别（0-正常、1-详细）
        - features_dim: 特征提取器的输出维度（默认 256）
        - **ppo_kwargs: 其他 PPO 参数（如 tensorboard_log）

    使用示例：
        # 基础用法
        model = create_ppo_model(env)

        # 自定义参数
        model = create_ppo_model(
            env,
            learning_rate=1e-4,
            features_dim=512,
            verbose=1,
            tensorboard_log="./logs"
        )
    """
    policy_kwargs = {
        "features_dim": features_dim,
        "latent_dim": int(latent_dim),
    }

    model = PPO(
        CustomPolicy,
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        **ppo_kwargs,
    )

    return model

import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv

ray.init()

args = argparse.Namespace
# args.env = "3DBall"
args.env = "CartPoleLeftRight"
args.file_name = None
args.from_checkpoint = None
args.as_test = True
args.stop_iters = 9999
args.stop_timesteps = 10000000
args.stop_reward = 9999.0
args.horizon = 3000
args.framework = "torch"
args.num_workers = 0

tune.register_env(
    "unity3d",
    lambda c: Unity3DEnv(
        file_name=c["file_name"],
        episode_horizon=c["episode_horizon"],
    ),
)



from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np
from typing import Callable, Tuple

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID, AgentID
def get_policy_configs_for_game(
        game_name: str,
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
    obs_spaces = {
        # 3DBall.
        "3DBall": Box(float("-inf"), float("inf"), (8,)),
        # CartPoleLeftRight.
        "CartPoleLeftRight": Box(float("-inf"), float("inf"), (5,)),
    }
    action_spaces = {
        # 3DBall.
        "3DBall": Box(-1.0, 1.0, (2,), dtype=np.float32),
        # CartPoleLeftRight.
        "CartPoleLeftRight": MultiDiscrete([3],),
    }
    # Policies (Unity: "behaviors") and agent-to-policy mapping fns.
    policies = {
        game_name: PolicySpec(
            observation_space=obs_spaces[game_name],
            action_space=action_spaces[game_name],
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return game_name
    return policies, policy_mapping_fn


policies, policy_mapping_fn = get_policy_configs_for_game(args.env)

config = (
    PPOConfig()
        .environment(
        env="unity3d",
        env_config={
            "file_name": args.file_name,
            "episode_horizon": args.horizon,
        },
        disable_env_checking=True,
    )
        .framework(args.framework)
        .rollouts(
        num_rollout_workers=args.num_workers if args.file_name else 0,
        rollout_fragment_length=200,
    )
        .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=256,
        train_batch_size=4000,
        num_sgd_iter=20,
        vf_loss_coeff=0.5,
        clip_param=0.2,
        entropy_coeff=0.05,
        model={"fcnet_hiddens": [64, 64]},

    )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

stop = {
    "training_iteration": args.stop_iters,
    "timesteps_total": args.stop_timesteps,
    "episode_reward_mean": args.stop_reward,
}

agent = PPO(config)
for _ in range(50):
    results = agent.train()
    print(results['episode_reward_mean'])

ray.shutdown()

from pettingzoo.classic import connect_four_v3
import ray.rllib.algorithms.ppo as ppo
import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_torch
from bnbot.wrappers.connect4wrapper import Connect4Env
from bnbot.models.connect4model import Connect4MaskModel

torch = try_import_torch()


def main():
    # define how to make the environment. This way takes an optional environment config, num_floors
    env_creator = lambda config: connect_four_v3.env(render_mode="rgb_array")

    # register that way to make the environment under an rllib name
    register_env("connect4", lambda config: Connect4Env(env_creator(config)))

    config = (
        ppo.PPOConfig()
        .environment("connect4")
        .framework("torch")
        .training(model={"custom_model": Connect4MaskModel})
        .multi_agent(
            policies={"policy_0", "policy_1"},
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kw: (
                    "policy_0" if agent_id == "player_0" else "policy_1"
                )
            ),
            policies_to_train=[
                "policy_0",
                "policy_1",
            ],
        )
    )

    stop = {
        "training_iteration": 10,
        "timesteps_total": 100000,
        "episode_reward_mean": 200,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )

    results = tuner.fit()


if __name__ == "__main__":
    main()

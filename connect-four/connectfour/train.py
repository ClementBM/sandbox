import argparse
import random

import ray
import ray.rllib.algorithms.ppo as ppo
from pettingzoo.classic import connect_four_v3
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch
from ray.tune import CLIReporter, register_env
from ray.rllib.algorithms.algorithm import Algorithm

from connectfour.callbacks import create_self_play_callback
from connectfour.dummy_policies import (
    AlwaysSameHeuristic,
    BeatLastHeuristic,
    LinearHeuristic,
    RandomHeuristic,
    SmartHeuristic,
)
from connectfour.models import Connect4MaskModel
from connectfour.conv_models import Connect4MaskConvModel
from connectfour.wrappers import Connect4Env
from ray.rllib.models import ModelCatalog

torch, nn = try_import_torch()


def get_cli_args():
    """
    Create CLI parser and return parsed arguments
    python connectfour/train.py --win-rate-threshold 0.4 --stop-iters 10 > training.log 2>&1
    python connectfour/train.py --num-cpus 5 --num-gpus 1 --win-rate-threshold 0.95 --stop-iters 10000 --stop-timesteps 1000000000 > training.log 2>&1
    python connectfour/train.py --num-cpus 5 --num-gpus 1 --win-rate-threshold 0.95 --stop-iters 15000 --from-checkpoint ~/ray_results/PPO/PPO_connect4_a999d_00000_0_2023-04-07_10-53-27/checkpoint_000500 > training.log 2>&1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Full path to a experiment directory to resume tuning from "
        "a previously saved Algorithm state.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=200, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=10000000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--win-rate-threshold",
        type=float,
        default=0.95,
        help="Win-rate at which we setup another opponent by freezing the "
        "current main policy and playing against a uniform distribution "
        "of previously frozen 'main's from here on.",
    )
    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


def save_policies(policy_ids, algo):
    for policy_id in policy_ids:
        print(f"Saving policy {policy_id}")
        ppo_policy = algo.get_policy(policy_id)

        # Save as torch model
        ppo_policy.export_model(f"models/{policy_id}")
        # Save as ONNX model
        ppo_policy.export_model(f"models/{policy_id}", onnx=11)


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(
        num_cpus=args.num_cpus or None,
        num_gpus=args.num_gpus,
        include_dashboard=True,
        resources={"accelerator_type:RTX": 1},
    )

    # define how to make the environment
    env_creator = lambda config: connect_four_v3.env(render_mode="rgb_array")

    # register that way to make the environment under an rllib name
    register_env("connect4", lambda config: Connect4Env(env_creator(config)))

    def select_policy(agent_id, episode, **kwargs):
        if episode.episode_id % 2 == int(agent_id[-1:]):
            return "learned"
        else:
            return random.choice(
                ["smart", "always_same", "beat_last", "random", "linear"]
            )

    ModelCatalog.register_custom_model("connect4conv", Connect4MaskConvModel)

    config = (
        (
            ppo.PPOConfig()
            .environment("connect4")
            .framework("torch")
            .training(
                model={
                    "custom_model": "connect4conv",
                    "post_fcnet_hiddens": [256, 256],
                    "conv_filters": [[32, [4, 4], 1]],
                    # Channel, [Kernel, Kernel], Stride]
                }
            )
            .rollouts(
                num_rollout_workers=args.num_workers,
                num_envs_per_worker=5,
            )
            # .checkpointing(checkpoint_trainable_policies_only=True)
        )
        .multi_agent(
            policies={
                "learned": PolicySpec(),
                "smart": PolicySpec(policy_class=SmartHeuristic),
                "always_same": PolicySpec(policy_class=AlwaysSameHeuristic),
                "linear": PolicySpec(policy_class=LinearHeuristic),
                "beat_last": PolicySpec(policy_class=BeatLastHeuristic),
                "random": PolicySpec(policy_class=RandomHeuristic),
            },
            policy_mapping_fn=select_policy,
            policies_to_train=["learned"],
        )
        .callbacks(
            create_self_play_callback(
                win_rate_thr=args.win_rate_threshold,
                opponent_policies=[
                    "smart",
                    "always_same",
                    "beat_last",
                    "random",
                    "linear",
                ],
                opponent_count=7,
            )
        )
        .evaluation(evaluation_interval=1)
    )

    if args.from_checkpoint is None:
        stop = {
            "timesteps_total": args.stop_timesteps,
            "training_iteration": args.stop_iters,
        }

        results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop,
                verbose=2,
                progress_reporter=CLIReporter(
                    metric_columns={
                        "training_iteration": "iter",
                        "time_total_s": "time_total_s",
                        "timesteps_total": "ts",
                        "episodes_this_iter": "train_episodes",
                        "policy_reward_mean/learned": "reward",
                        "win_rate": "win_rate",
                        "league_size": "league_size",
                    },
                    mode="max",
                    metric="win_rate",
                    sort_by_metric=True,
                ),
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=10,
                    checkpoint_at_end=True,
                    checkpoint_frequency=10,
                    checkpoint_score_order="max",
                ),
            ),
        ).fit()

        best_checkpoint = results.get_best_result(
            metric="win_rate", mode="max"
        ).checkpoint
        print("Best checkpoint", best_checkpoint)

        algo = Algorithm.from_checkpoint(checkpoint=best_checkpoint)
        save_policies(list(algo.workers.local_worker().policy_map.keys()), algo)

    else:
        algo = Algorithm.from_checkpoint(checkpoint=args.from_checkpoint)

        config = algo.config.copy(False)
        config.checkpointing(export_native_model_files=True)

        opponent_policies = list(algo.workers.local_worker().policy_map.keys())
        opponent_policies.remove("learned")
        opponent_policies.sort()

        save_policies(opponent_policies, algo)

        config.callbacks(
            create_self_play_callback(
                win_rate_thr=args.win_rate_threshold,
                opponent_policies=opponent_policies,
                opponent_count=len(opponent_policies),
            )
        )
        config.evaluation(evaluation_interval=None)

        analysis = tune.run(
            "PPO",
            config=config.to_dict(),
            restore=args.from_checkpoint,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            keep_checkpoints_num=10,
            mode="max",
            metric="win_rate",
            stop={
                "win_rate": args.win_rate_threshold,
                "training_iteration": args.stop_iters,
            },
            progress_reporter=CLIReporter(
                metric_columns={
                    "training_iteration": "iter",
                    "time_total_s": "time_total_s",
                    "timesteps_total": "ts",
                    "episodes_this_iter": "train_episodes",
                    "policy_reward_mean/learned": "reward",
                    "win_rate": "win_rate",
                    "league_size": "league_size",
                }
            ),
        )

        algo = Algorithm.from_checkpoint(analysis.best_checkpoint)
        ppo_policy = algo.get_policy("learned")

        # Save as torch model
        ppo_policy.export_model("models")
        # Save as ONNX model
        ppo_policy.export_model("models", onnx=11)

        print("Best checkpoint", analysis.best_checkpoint)

    ray.shutdown()

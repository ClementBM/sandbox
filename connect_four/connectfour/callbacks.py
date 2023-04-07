from collections import deque

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from connectfour.dummy_policies import (
    AlwaysSameHeuristic,
    BeatLastHeuristic,
    LinearHeuristic,
    RandomHeuristic,
    SmartHeuristic,
)


def create_self_play_callback(win_rate_thr, opponent_policies, opponent_count=10):
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.current_opponent = 0
            self.opponent_policies = deque(opponent_policies, maxlen=opponent_count)
            self.policy_to_remove = None
            self.win_rate_threshold = win_rate_thr
            self.frozen_policies = {
                # "always_same": AlwaysSameHeuristic,
                # "linear": LinearHeuristic,
                # "beat_last": BeatLastHeuristic,
                # "random": RandomHeuristic,
                "smart": SmartHeuristic
            }

        def on_train_result(self, *, algorithm, result, **kwargs):
            """Called at the end of Algorithm.train().

            Args:
                algorithm: Current Algorithm instance.
                result: Dict of results returned from Algorithm.train() call.
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
            """
            main_rew = result["hist_stats"].pop("policy_learned_reward")
            opponent_rew = result["hist_stats"].pop("episode_reward")

            won = 0
            for r_main, r_opponent in zip(main_rew, opponent_rew):
                if r_main > r_opponent:
                    won += 1
            win_rate = won / len(main_rew)

            result["win_rate"] = win_rate
            print(f"Iter={algorithm.iteration} win-rate={win_rate}")

            if win_rate > self.win_rate_threshold:
                if len(self.opponent_policies) == self.opponent_policies.maxlen:
                    self.policy_to_remove = self.opponent_policies[0]

                new_pol_id = None
                while new_pol_id is None:
                    if np.random.choice(range(3)) == 0:
                        new_pol_id = np.random.choice(list(self.frozen_policies.keys()))
                    else:
                        self.current_opponent += 1
                        new_pol_id = f"learned_v{self.current_opponent}"

                    if new_pol_id in self.opponent_policies:
                        new_pol_id = None
                    else:
                        self.opponent_policies.append(new_pol_id)

                print("Non trainable policies", list(self.opponent_policies))

                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    return (
                        "learned"
                        if episode.episode_id % 2 == int(agent_id[-1:])
                        else np.random.choice(list(self.opponent_policies))
                    )

                print(
                    f"Iter={algorithm.iteration} Adding new opponent to the mix ({new_pol_id}). League size {len(self.opponent_policies) + 1}"
                )

                if new_pol_id in list(self.frozen_policies.keys()):
                    new_policy = algorithm.add_policy(
                        policy_id=new_pol_id,
                        policy_cls=self.frozen_policies[new_pol_id],
                        policy_mapping_fn=policy_mapping_fn,
                    )
                else:
                    new_policy = algorithm.add_policy(
                        policy_id=new_pol_id,
                        policy_cls=type(algorithm.get_policy("learned")),
                        policy_mapping_fn=policy_mapping_fn,
                    )
                    learned_state = algorithm.get_policy("learned").get_state()
                    new_policy.set_state(learned_state)
                algorithm.workers.sync_weights()

            else:
                print("Not good enough... Keep learning ...")

            result["league_size"] = len(self.opponent_policies) + 1

        def on_evaluate_end(self, *, algorithm, evaluation_metrics, **kwargs):
            """Runs when the evaluation is done.

            Runs at the end of Algorithm.evaluate().

            Args:
                algorithm: Reference to the algorithm instance.
                evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
            """

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                return (
                    "learned"
                    if episode.episode_id % 2 == int(agent_id[-1:])
                    else np.random.choice(list(self.opponent_policies))
                )

            if self.policy_to_remove is not None:
                print("Remove ", self.policy_to_remove, "from opponent policies")
                algorithm.remove_policy(
                    self.policy_to_remove,
                    policy_mapping_fn=policy_mapping_fn,
                )
                self.policy_to_remove = None
                algorithm.workers.sync_weights()

    return SelfPlayCallback

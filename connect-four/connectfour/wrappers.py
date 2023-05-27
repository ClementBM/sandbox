from typing import Optional

from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


@PublicAPI
class Connect4Env(PettingZooEnv):
    """An interface to the PettingZoo MARL environment library"""

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # In base class =>
        # info = self.env.reset(seed=seed, return_info=True, options=options)
        info = self.env.reset(seed=seed, options=options)
        return (
            {self.env.agent_selection: self.env.observe(self.env.agent_selection)},
            info or {},
        )

    def render(self):
        # In base class =>
        # return self.env.render(self.render_mode)
        return self.env.render()

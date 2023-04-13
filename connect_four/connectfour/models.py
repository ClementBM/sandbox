from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gymnasium.spaces import Dict
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.utils import override
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

torch, nn = try_import_torch()


class Connect4MaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)

        assert isinstance(orig_space, Dict)
        assert "action_mask" in orig_space.spaces
        assert "observation" in orig_space.spaces

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


# class SacConnect4MaskModel(SACTorchModel):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name: str,
#         policy_model_config=None,
#         q_model_config=None,
#         twin_q=False,
#         initial_alpha=1.0,
#         target_entropy=None,
#         **kwargs,
#     ):
#         orig_space = getattr(obs_space, "original_space", obs_space)

#         assert isinstance(orig_space, Dict)
#         assert "action_mask" in orig_space.spaces
#         assert "observation" in orig_space.spaces

#         super().__init__(
#             obs_space,
#             action_space,
#             num_outputs,
#             model_config,
#             policy_model_config,
#             q_model_config,
#             twin_q,
#             initial_alpha,
#             target_entropy,
#             **kwargs,
#         )

#         self.internal_model = FullyConnectedNetwork(
#             orig_space["observation"],
#             action_space,
#             num_outputs,
#             model_config,
#             name + "_internal",
#         )

#     @override(SACTorchModel)
#     def forward(self, input_dict, state, seq_lens):
#         # Extract the available actions tensor from the observation.
#         action_mask = input_dict["obs"]["action_mask"]

#         # Compute the unmasked logits.
#         logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

#         # Convert action_mask into a [0.0 || -inf]-type mask.
#         inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
#         masked_logits = logits + inf_mask

#         # Return masked logits.
#         return masked_logits, state

#     def value_function(self):
#         return self.internal_model.value_function()

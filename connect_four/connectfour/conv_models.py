import numpy as np
from gymnasium.spaces import Dict, Box
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()


class Connect4MaskConvModel(TorchModelV2, nn.Module):
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
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        self._logits = None

        layers = []
        (w, h, in_channels) = orig_space["observation"].shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.

        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )

        # num_outputs defined. Use that to create an exact
        # `num_output`-sized (1,1)-Conv2D.
        in_size = [
            np.ceil((in_size[0] - kernel[0]) / stride),
            np.ceil((in_size[1] - kernel[1]) / stride),
        ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])

        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
            layers.append(
                SlimFC(
                    in_size=in_size * 12 if i == 0 else in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation
                    if i < len(post_fcnet_hiddens) - 1
                    else None,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size
        # Last layer is logits layer.
        self._logits = layers.pop()

        self._convs = nn.Sequential(*layers)

        # Value function
        vf_layers = []
        (w, h, in_channels) = orig_space["observation"].shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        vf_layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,
                activation_fn=activation,
            )
        )

        vf_layers.append(
            SlimConv2d(
                in_channels=out_channels,
                out_channels=1,
                kernel=(3, 4),
                stride=1,
                padding=None,
                activation_fn=None,
            )
        )
        self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        self._features = input_dict["obs"]["observation"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)

        logits = None
        if self._logits:
            conv_out = self._logits(conv_out)
        if len(conv_out.shape) == 4:
            if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                raise ValueError(
                    "Given `conv_filters` ({}) do not result in a [B, {} "
                    "(`num_outputs`), 1, 1] shape (but in {})! Please "
                    "adjust your Conv2D stack such that the last 2 dims "
                    "are both 1.".format(
                        self.model_config["conv_filters"],
                        self.num_outputs,
                        list(conv_out.shape),
                    )
                )
            logits = conv_out.squeeze(3)
            logits = logits.squeeze(2)
        else:
            logits = conv_out

        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

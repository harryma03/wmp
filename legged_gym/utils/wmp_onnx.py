"""Deployment exports for the WMP actor and one world-model observation step."""

import copy
import os

import torch
from torch import nn


class WMPPolicy(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
        self.wm_feature_encoder = copy.deepcopy(actor_critic.wm_feature_encoder)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.command_start = actor_critic.privileged_dim + 6

    def forward(self, observation, history, wm_feature):
        latent = self.history_encoder(history)
        command = observation[:, self.command_start:self.command_start + 3]
        wm_latent = self.wm_feature_encoder(wm_feature)
        return self.actor(torch.cat((latent, command, wm_latent), dim=-1))


class WMPWorldModelStep(nn.Module):
    """One deterministic RSSM update; caller retains the returned state."""

    def __init__(self, world_model):
        super().__init__()
        self.encoder = copy.deepcopy(world_model.encoder)
        self.dynamics = copy.deepcopy(world_model.dynamics)
        if not self.dynamics._discrete:
            raise ValueError("WMP ONNX export currently requires a discrete RSSM")

    def forward(self, prop, image, prev_stoch, prev_deter, action_history, is_first):
        dynamics = self.dynamics
        reset = is_first.reshape(-1, 1)
        initial_deter = torch.tanh(dynamics.W).expand_as(prev_deter)
        initial_stoch = dynamics.get_stoch(initial_deter)
        stoch = prev_stoch * (1 - reset.unsqueeze(-1)) + initial_stoch * reset.unsqueeze(-1)
        deter = prev_deter * (1 - reset) + initial_deter * reset
        action = action_history * (1 - reset)

        embed = self.encoder({"prop": prop, "image": image})
        prior = dynamics.img_step({"stoch": stoch, "deter": deter}, action, sample=False)
        x = dynamics._obs_out_layers(torch.cat((prior["deter"], embed), dim=-1))
        logits = dynamics._suff_stats_layer("obs", x)["logit"]
        posterior = torch.nn.functional.one_hot(logits.argmax(dim=-1), dynamics._discrete).to(logits.dtype)
        return posterior, prior["deter"]


def export_wmp_as_onnx(actor_critic, world_model, path, obs_dim, history_dim,
                       prop_dim, image_shape, action_dim, action_interval):
    """Write policy.onnx and world_model.onnx with dynamic batch dimensions."""
    try:
        import onnx  # noqa: F401 - required by the PyTorch exporter
    except ImportError as exc:
        raise RuntimeError("ONNX export requires `pip install onnx`") from exc

    os.makedirs(path, exist_ok=True)
    policy = WMPPolicy(actor_critic).cpu().eval()
    wm = WMPWorldModelStep(world_model).cpu().eval()
    dynamics = wm.dynamics
    batch = 2
    with torch.no_grad():
        policy_inputs = (torch.zeros(batch, obs_dim), torch.zeros(batch, history_dim),
                         torch.zeros(batch, dynamics._deter))
        torch.onnx.export(
            policy, policy_inputs, os.path.join(path, "policy.onnx"),
            input_names=["observation", "history", "wm_feature"], output_names=["action"],
            dynamic_axes={name: {0: "batch"} for name in
                          ("observation", "history", "wm_feature", "action")},
            opset_version=15, dynamo=False)

        wm_inputs = (torch.zeros(batch, prop_dim), torch.zeros((batch,) + tuple(image_shape)),
                     torch.zeros(batch, dynamics._stoch, dynamics._discrete),
                     torch.zeros(batch, dynamics._deter),
                     torch.zeros(batch, action_dim * action_interval), torch.zeros(batch))
        names = ["prop", "image", "prev_stoch", "prev_deter", "action_history", "is_first"]
        outputs = ["next_stoch", "wm_feature"]
        torch.onnx.export(
            wm, wm_inputs, os.path.join(path, "world_model.onnx"),
            input_names=names, output_names=outputs,
            dynamic_axes={name: {0: "batch"} for name in names + outputs},
            opset_version=15, dynamo=False)
    return path

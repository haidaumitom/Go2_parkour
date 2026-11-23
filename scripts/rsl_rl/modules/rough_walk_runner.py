"""Load pre-waypoint rough-walk policies into the waypoint-aware runner."""

from copy import deepcopy

import torch
from rsl_rl.runners import OnPolicyRunner
from torch import nn


class GaitPreservingActor(nn.Module):
    """Blend from the frozen walking actor to the fine-tuned actor near a hurdle."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.adapted = actor
        self.reference = deepcopy(actor).requires_grad_(False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hurdle_xy = observations[..., -3:-1]
        hurdle_height = observations[..., -1:]
        distance = hurdle_xy.norm(dim=-1, keepdim=True)
        # Fade between the two actors over the approach and landing on either side of the hurdle.
        blend = ((2.0 - distance) / (2.0 - 0.75)).clamp(0.0, 1.0)
        blend = blend.square() * (3.0 - 2.0 * blend) * (hurdle_height > 0)
        return torch.lerp(self.reference(observations), self.adapted(observations), blend)


class RoughWalkOnPolicyRunner(OnPolicyRunner):
    """Warm-start waypoint policies from the 48-input walking checkpoints."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if train_cfg.get("preserve_walking_gait"):
            policy = self.alg.policy
            policy.actor = GaitPreservingActor(policy.actor)
            noise = policy.std if policy.noise_std_type == "scalar" else policy.log_std
            noise.requires_grad_(False)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        previous = checkpoint["model_state_dict"]
        current = self.alg.policy.state_dict()
        preserve_gait = isinstance(self.alg.policy.actor, GaitPreservingActor)
        actor_weights = "actor.adapted.0.weight" if preserve_gait else "actor.0.weight"
        input_layers = {"actor.0.weight": actor_weights, "critic.0.weight": "critic.0.weight"}
        is_pre_waypoint = all(
            name in previous
            and target in current
            and previous[name].ndim == current[target].ndim == 2
            and previous[name].shape[1] == 48
            and current[target].shape == (previous[name].shape[0], 51)
            for name, target in input_layers.items()
        )
        if not is_pre_waypoint:
            del previous, checkpoint
            return super().load(path, load_optimizer=load_optimizer)

        if self.empirical_normalization or self.alg.rnd:
            raise ValueError("Pre-waypoint rough-walk checkpoints require observation normalization and RND to be disabled.")

        # The original observation channels retain their order. Start with zero
        # weights on the three appended channels to preserve the learned gait.
        for name in input_layers:
            previous[name] = torch.nn.functional.pad(previous[name], (0, 3))
        if preserve_gait:
            for name in tuple(previous):
                if name.startswith("actor."):
                    layer = name.removeprefix("actor.")
                    previous[f"actor.reference.{layer}"] = previous[f"actor.adapted.{layer}"] = previous.pop(name)
        self.alg.policy.load_state_dict(previous)
        self.alg.optimizer.state.clear()
        self.current_learning_iteration = 0
        print("[INFO] Warm-started waypoint policy from walking weights with a fresh optimizer at iteration 0.")
        return checkpoint.get("infos")

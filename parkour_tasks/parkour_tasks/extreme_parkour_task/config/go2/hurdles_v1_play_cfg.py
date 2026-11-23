"""Playback environment for the original five-hurdle checkpoint."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.utils import configclass

from parkour_tasks.default_cfg import VIEWER

from .rough_walk_cfg import RoughWalkClearanceVelocityCommand, RoughnessLevels, UnitreeGo2RoughWalkEnvCfg


class HurdlesV1CourseEnd(ManagerTermBase):
    """Use the original course rule based on visits to all hurdle footprints."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        generator = env.scene.terrain.cfg.terrain_generator
        terrain = generator.sub_terrains["rough_walk"]
        self.positions = torch.as_tensor(terrain.hurdle_positions, device=env.device, dtype=torch.float32).reshape(-1, 2)
        self.visited = torch.zeros((env.num_envs, len(self.positions)), device=env.device, dtype=torch.bool)
        self.completed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.half_depth = terrain.hurdle_depth / 2
        self.half_width = terrain.hurdle_width / 2
        if len(self.positions):
            last_edge = max(position[0] for position in terrain.hurdle_positions) + self.half_depth
            self.finish_x = min(last_edge + 1.0, generator.size[0] / 2)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.visited[ids] = False
        self.completed[ids] = False

    def __call__(self, env):
        if not len(self.positions):
            return self.completed
        xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        delta = (xy[:, None, :] - self.positions).abs()
        self.visited |= (delta[..., 0] <= self.half_depth) & (delta[..., 1] <= self.half_width)
        at_finish = xy[:, 0] >= self.finish_x
        self.completed.copy_(at_finish & self.visited.all(dim=1))
        return at_finish


class HurdlesV1RoughnessLevels(RoughnessLevels):
    def _episode_completed(self, env, command):
        course = env.termination_manager.get_term_cfg("hurdle_course_end").func
        return course.completed if len(course.positions) else env.termination_manager.time_outs


@configclass
class UnitreeGo2HurdlesV1PlayEnvCfg(UnitreeGo2RoughWalkEnvCfg):
    """Restore the training settings saved with walk_jump_hurdles_v1."""

    viewer = VIEWER.replace(eye=(5.0, 13.0, 9.0), lookat=(8.0, 0.0, 0.0))

    def __post_init__(self):
        super().__post_init__()
        self.seed = 1
        terrain = self.scene.terrain.terrain_generator.sub_terrains["rough_walk"]
        terrain.hurdle_positions = [[1.5 + 5.0 * i, 0.0] for i in range(5)]
        terrain.hurdle_height = 0.05
        terrain.hurdle_height_increase = 0.10

        command = self.commands.base_velocity
        command.class_type = RoughWalkClearanceVelocityCommand
        command.ranges.lin_vel_x = (0.8, 1.2)
        self.observations.policy.hurdle_waypoint = None

        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp
        self.rewards.track_ang_vel_z_exp = self.rewards.track_heading.replace(func=mdp.track_ang_vel_z_exp)
        self.rewards.track_heading = None
        self.rewards.waypoint_velocity = None
        self.rewards.route_deviation = None
        self.rewards.hurdle_crossing = None

        self.terminations.hurdle_course_end = TerminationTermCfg(func=HurdlesV1CourseEnd, time_out=True)
        self.terminations.hurdle_waypoint_missed = None
        self.curriculum.terrain_levels.func = HurdlesV1RoughnessLevels

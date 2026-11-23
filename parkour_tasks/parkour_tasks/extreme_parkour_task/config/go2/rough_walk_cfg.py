"""Forward velocity tracking on nominally level terrain with optional hurdles."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import (
    CurriculumTermCfg,
    ManagerTermBase,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from parkour_isaaclab.envs.mdp.rewards import (
    JointTargetSmoothness,
    estimate_ground_height,
    foot_clearance_penalty,
    joint_power,
)


FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def walking_surface(difficulty: float, cfg: RoughWalkTerrainCfg):
    """Generate rough ground and add hurdles in tile-relative coordinates."""
    amplitude = 0.005 + 0.030 * difficulty
    meshes, origin = random_uniform_terrain(difficulty, cfg.replace(noise_range=(-amplitude, amplitude)))
    if not cfg.hurdle_positions:
        return meshes, origin

    positions = np.asarray(cfg.hurdle_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError("hurdle_positions must contain finite [forward_m, lateral_m] pairs")

    dimensions = np.asarray([cfg.hurdle_depth, cfg.hurdle_width, cfg.hurdle_height], dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("hurdle_depth, hurdle_width and hurdle_height must be finite positive numbers")
    if not np.isfinite(cfg.hurdle_height_increase) or cfg.hurdle_height_increase < 0:
        raise ValueError("hurdle_height_increase must be finite and non-negative")
    height = dimensions[2] + cfg.hurdle_height_increase * np.clip(difficulty, 0.0, 1.0)
    footprint = dimensions[:2]
    half_footprint = footprint / 2

    # Leave space for the robot and its reset position jitter around the tile center.
    if np.any(positions[:, 0] - half_footprint[0] < 1.0):
        raise ValueError("the near edge of each hurdle must be at least 1 m ahead of the tile center")
    if np.any(np.abs(positions) + half_footprint > np.asarray(cfg.size) / 2):
        raise ValueError("each hurdle footprint must fit within its terrain tile")

    ground_vertices = meshes[0].vertices
    for index, position in enumerate(positions):
        if np.any(np.all(np.abs(positions[:index] - position) < footprint, axis=1)):
            raise ValueError("hurdle footprints must not overlap")
        center = origin[:2] + position
        # Include the adjacent height samples so the footing is supported even on narrow bars.
        nearby = np.all(np.abs(ground_vertices[:, :2] - center) <= half_footprint + cfg.horizontal_scale, axis=1)
        ground_z = ground_vertices[nearby, 2]
        bottom = ground_z.min() - cfg.vertical_scale
        top = ground_z.max() + height
        hurdle = trimesh.creation.box(extents=[*footprint, top - bottom])
        hurdle.apply_translation([*center, (bottom + top) / 2])
        meshes.append(hurdle)
    return meshes, origin


@configclass
class RoughWalkTerrainCfg(HfRandomUniformTerrainCfg):
    function = walking_surface
    hurdle_positions: list[list[float]] = []
    """Hurdle centers as [forward, lateral] offsets in meters from the tile center."""
    hurdle_height: float = 0.05
    """Height above the surrounding ground at the easiest difficulty, in meters."""
    hurdle_height_increase: float = 0.0
    """Additional hurdle height at the hardest difficulty, in meters."""
    hurdle_depth: float = 0.20
    """Thickness in the walking direction, in meters."""
    hurdle_width: float = 2.0
    """Width across the lane, in meters."""
    finish_distance: float = 1.0
    """Distance from the final hurdle's far edge to the finish line, in meters."""


def base_height_l2_finite(env, target_height: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """Penalize squared clearance error using only valid ground hits."""
    ground_height, ground_valid = estimate_ground_height(env.scene.sensors[sensor_cfg.name].data.ray_hits_w)
    clearance = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - ground_height
    penalty = (clearance - target_height).square()
    return torch.where(ground_valid & torch.isfinite(penalty), penalty, 0.0)


class HurdleWaypoints:
    """Track an ordered route with one waypoint beyond each hurdle."""

    lateral_tolerance = 0.35
    max_overshoot = 0.40

    def __init__(self, env):
        generator = env.scene.terrain.cfg.terrain_generator
        terrain = generator.sub_terrains["rough_walk"]
        positions = sorted(terrain.hurdle_positions, key=lambda point: point[0])
        self.count = len(positions)
        self.half_depth = terrain.hurdle_depth / 2
        self.half_width = terrain.hurdle_width / 2
        self.positions = torch.tensor(positions + [[0.0, 0.0]], device=env.device, dtype=torch.float32)
        self.waypoints = self.positions.clone()
        if positions:
            last_edge = positions[-1][0] + self.half_depth
            finish_x = min(last_edge + terrain.finish_distance, generator.size[0] / 2)
            self.waypoints[:-1, 0] += self.half_depth + 0.5
            self.waypoints[:, 0].clamp_max_(finish_x)
            self.waypoints[-1, 0] = finish_x
            self.waypoints[-1, 1] = positions[-1][1]
        self.route = torch.cat((torch.zeros_like(self.waypoints[:1]), self.waypoints), dim=0)
        self.index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.entered_bar = torch.zeros_like(self.index, dtype=torch.bool)
        self.crossed_bar = torch.zeros_like(self.entered_bar)
        self.completed = torch.zeros_like(self.entered_bar)

    def reset(self, ids):
        self.index[ids] = 0
        self.entered_bar[ids] = False
        self.crossed_bar[ids] = False
        self.completed[ids] = False

    def update(self, xy):
        if self.count == 0:
            return self.completed

        active = self.index < self.count
        offset = xy - self.positions[self.index]
        in_lane = offset[:, 1].abs() <= self.half_width
        self.entered_bar |= active & in_lane & (offset[:, 0].abs() <= self.half_depth)
        on_target = offset[:, 1].abs() <= min(self.half_width, self.lateral_tolerance)
        self.crossed_bar.copy_(active & self.entered_bar & on_target & (xy[:, 0] >= self.waypoints[self.index, 0]))
        self.index.add_(self.crossed_bar)
        self.entered_bar &= ~self.crossed_bar
        at_finish = xy[:, 0] >= self.waypoints[-1, 0]
        on_finish = (xy[:, 1] - self.waypoints[-1, 1]).abs() <= self.lateral_tolerance
        self.completed.copy_(at_finish & on_finish & (self.index == self.count))
        return at_finish

    def lateral_error(self, xy):
        start = self.route[self.index]
        end = self.route[self.index + 1]
        fraction = ((xy[:, 0] - start[:, 0]) / (end[:, 0] - start[:, 0]).clamp_min(1e-5)).clamp(0.0, 1.0)
        return (xy[:, 1] - torch.lerp(start[:, 1], end[:, 1], fraction)).abs()


class RoughWalkClearanceVelocityCommand(mdp.UniformVelocityCommand):
    """Track mean measured base clearance alongside command tracking."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.height_scanner = env.scene.sensors["height_scanner"]
        self.clearance_samples = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_clearance_m"] = torch.zeros_like(self.clearance_samples)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.clearance_samples[ids] = 0
        return super().reset(env_ids)

    def _update_metrics(self):
        super()._update_metrics()
        self._update_clearance_metrics()

    def _update_clearance_metrics(self):
        ground_height, ground_valid = estimate_ground_height(self.height_scanner.data.ray_hits_w)
        clearance = self.robot.data.root_pos_w[:, 2] - ground_height
        valid = ground_valid & torch.isfinite(clearance) & (self._env.episode_length_buf > 0)
        self.clearance_samples.add_(valid)
        mean_clearance = self.metrics["base_clearance_m"]
        mean_clearance.add_(torch.where(valid, clearance - mean_clearance, 0.0) / self.clearance_samples.clamp_min(1))


class RoughWalkVelocityCommand(RoughWalkClearanceVelocityCommand):
    """Track velocity along the active hurdle route."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.course = HurdleWaypoints(env)
        self.metrics["hurdles_crossed"] = torch.zeros_like(self.clearance_samples)
        self.metrics["route_error_m"] = torch.zeros_like(self.clearance_samples)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.course.reset(ids)
        return super().reset(env_ids)

    def position_xy(self):
        return self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]

    def goal_offset_w(self):
        return self.course.waypoints[self.course.index] - self.position_xy()

    def goal_direction_w(self):
        return torch.nn.functional.normalize(self.goal_offset_w(), dim=1, eps=1e-5)

    def velocity_error_xy(self):
        target_velocity = self.goal_direction_w() * self.command[:, :1]
        return target_velocity - self.robot.data.root_lin_vel_w[:, :2]

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "waypoint_visualizer"):
            marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/hurdle_waypoint")
            sphere = marker_cfg.markers["sphere"]
            marker_cfg.markers = {
                "upcoming": sphere.replace(radius=0.08),
                "active": sphere.replace(
                    radius=0.12,
                    visual_material=sphere.visual_material.replace(diffuse_color=(0.1, 1.0, 0.1)),
                ),
                "path_dot": sphere.replace(radius=0.035),
            }
            self.waypoint_visualizer = VisualizationMarkers(marker_cfg)
            self._path_fractions = torch.linspace(0.15, 0.85, 5, device=self.device)[:, None]
        if hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized or not hasattr(self, "course") or not self.course.count:
            return
        # Show the route for the viewer's robot so markers do not overlap on shared tiles.
        origin = self._env.scene.env_origins[0]
        active = self.course.index[0]
        indices = torch.arange(self.course.count + 1, device=self.device)
        visible = indices >= active
        waypoints = origin.expand(self.course.count + 1, 3).clone()
        waypoints[:, :2] += self.course.waypoints
        waypoints[:, 2] += 0.15

        start = waypoints[0].clone()
        start[:2] = origin[:2] + self.course.route[active]
        dots = torch.lerp(start, waypoints[self.course.index[:1]], self._path_fractions)
        dot_indices = torch.full((dots.shape[0],), 2, device=self.device, dtype=torch.long)
        marker_indices = torch.cat(((indices[visible] == active).long(), dot_indices))
        self.waypoint_visualizer.visualize(torch.cat((waypoints[visible], dots)), marker_indices=marker_indices)

    def _update_metrics(self):
        if self.course.count:
            scale = self._env.step_dt / self.cfg.resampling_time_range[1]
            self.metrics["error_vel_xy"] += self.velocity_error_xy().norm(dim=1) * scale
            self.metrics["error_vel_yaw"] += (self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]).abs() * scale
            steps = self._env.episode_length_buf
            route_error = self.metrics["route_error_m"]
            error = self.course.lateral_error(self.position_xy())
            route_error.add_(torch.where(steps > 0, error - route_error, 0.0) / steps.clamp_min(1))
            self._update_clearance_metrics()
        else:
            super()._update_metrics()


@configclass
class RoughWalkVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = RoughWalkVelocityCommand


def hurdle_course_end(env):
    command = env.command_manager.get_term("base_velocity")
    finished = command.course.update(command.position_xy())
    command.metrics["hurdles_crossed"].copy_(command.course.index)
    return finished


def hurdle_waypoint_missed(env):
    command = env.command_manager.get_term("base_velocity")
    course = command.course
    return (course.index < course.count) & (command.goal_offset_w()[:, 0] < -course.max_overshoot)


def hurdle_waypoint_observation(env, nearest_hurdle: bool = False):
    """Report the waypoint or nearest hurdle in the robot's yaw frame, with its nominal height."""
    command = env.command_manager.get_term("base_velocity")
    if not command.course.count:
        return torch.zeros_like(command.command)
    if nearest_hurdle:
        offsets = command.course.positions[:-1] - command.position_xy()[:, None, :]
        nearest = offsets.square().sum(dim=2).argmin(dim=1)
        offset = offsets.gather(1, nearest[:, None, None].expand(-1, 1, 2)).squeeze(1)
    else:
        offset = command.goal_offset_w()
    offset = torch.cat((offset, torch.zeros_like(offset[:, :1])), dim=1)
    body_xy = quat_apply_inverse(yaw_quat(command.robot.data.root_quat_w), offset)[:, :2].clamp(-6.0, 6.0)
    importer = env.scene.terrain
    generator = importer.cfg.terrain_generator
    terrain = generator.sub_terrains["rough_walk"]
    low, high = generator.difficulty_range
    # Isaac Lab samples difficulty within each row. Use the row midpoint as the height estimate.
    difficulty = low + (high - low) * (importer.terrain_levels + 0.5) / generator.num_rows
    height = terrain.hurdle_height + terrain.hurdle_height_increase * difficulty.clamp(0.0, 1.0)
    if not nearest_hurdle:
        height = height * (command.course.index < command.course.count)
    return torch.cat((body_xy, height[:, None]), dim=1)


def waypoint_velocity_reward(env):
    command = env.command_manager.get_term("base_velocity")
    if not command.course.count:
        return torch.zeros_like(command.command[:, 0])
    progress_speed = (command.goal_direction_w() * command.robot.data.root_lin_vel_w[:, :2]).sum(dim=1)
    return (progress_speed / command.command[:, 0].clamp_min(0.1)).clamp(-1.0, 1.0)


def track_route_velocity(env, command_name: str, std: float):
    command = env.command_manager.get_term(command_name)
    if not command.course.count:
        return mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std)
    return torch.exp(-command.velocity_error_xy().square().sum(dim=1) / std**2)


def track_route_heading(env, command_name: str, std: float):
    command = env.command_manager.get_term(command_name)
    if not command.course.count:
        return mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std)
    goal = command.goal_offset_w()
    heading = torch.atan2(goal[:, 1], goal[:, 0])
    error = wrap_to_pi(heading - command.robot.data.heading_w)
    return torch.exp(-error.square() / std**2)


def route_deviation_penalty(env):
    command = env.command_manager.get_term("base_velocity")
    if not command.course.count:
        return torch.zeros_like(command.command[:, 0])
    return command.course.lateral_error(command.position_xy()).clamp_max(1.0).square()


def hurdle_crossing_bonus(env):
    command = env.command_manager.get_term("base_velocity")
    # The reward manager multiplies by step_dt. Give this bonus once per bar.
    return (command.course.crossed_bar & ~env.termination_manager.terminated).float() / env.step_dt


def hurdle_course_completion_bonus(env):
    command = env.command_manager.get_term("base_velocity")
    return (command.course.completed & ~env.termination_manager.terminated).float() / env.step_dt


class RoughnessLevels(ManagerTermBase):
    """Advance after consecutive successful rollouts with good speed and posture."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        self.success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _episode_completed(self, env, command):
        course = command.course
        return course.completed if course.count else env.termination_manager.time_outs

    def __call__(
        self,
        env,
        env_ids,
        promote_error: float,
        demote_error: float,
        min_base_clearance: float,
        successes_per_level: int,
    ):
        terrain = env.scene.terrain
        command = env.command_manager.get_term("base_velocity")
        episode_steps = env.episode_length_buf[env_ids]
        command_steps = command.cfg.resampling_time_range[1] / env.step_dt
        mean_error = command.metrics["error_vel_xy"][env_ids] * command_steps / episode_steps.clamp_min(1)
        completed = self._episode_completed(env, command)
        failed = env.termination_manager.terminated[env_ids]
        success = (
            completed[env_ids]
            & ~failed
            & (mean_error < promote_error)
            & (command.metrics["base_clearance_m"][env_ids] >= min_base_clearance)
        )
        streak = torch.where(success, self.success_streak[env_ids] + 1, 0)
        ready = streak >= successes_per_level
        move_up = ready & (terrain.terrain_levels[env_ids] < terrain.max_terrain_level - 1)
        move_down = failed | (mean_error > demote_error)
        self.success_streak[env_ids] = torch.where(ready | move_down, 0, streak)
        terrain.update_env_origins(env_ids, move_up, move_down)
        return terrain.terrain_levels.float().mean()


@configclass
class RoughWalkRewardsCfg:
    """Velocity tracking and healthy-leg gait rewards for rough walking."""

    track_lin_vel_xy_exp = RewardTermCfg(
        func=track_route_velocity, weight=1.5, params={"command_name": "base_velocity", "std": 0.25}
    )
    track_heading = RewardTermCfg(
        func=track_route_heading, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    waypoint_velocity = RewardTermCfg(func=waypoint_velocity_reward, weight=0.25)
    route_deviation = RewardTermCfg(func=route_deviation_penalty, weight=-0.5)
    hurdle_crossing = RewardTermCfg(func=hurdle_crossing_bonus, weight=1.0)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewardTermCfg(
        func=base_height_l2_finite,
        weight=-10.0,
        params={
            "target_height": 0.36,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    hip_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    leg_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    joint_power = RewardTermCfg(func=joint_power, weight=-2.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewardTermCfg(
        func=JointTargetSmoothness, weight=-0.01, params={"action_name": "joint_pos"}
    )
    feet_air_time = RewardTermCfg(
        func=velocity_mdp.feet_air_time,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES),
            "threshold": 0.20,
        },
    )
    feet_slide = RewardTermCfg(
        func=velocity_mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
        },
    )
    foot_clearance = RewardTermCfg(
        func=foot_clearance_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
            "target_height": 0.12,
            "tanh_mult": 1.0,
        },
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )


@configclass
class UnitreeGo2RoughWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.rewards = RoughWalkRewardsCfg()
        self.scene.num_envs = 4096
        # Upper limit for rollouts that do not reach the finish.
        self.episode_length_s = 80.0
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(50.0, 6.0),
            border_width=5.0,
            num_rows=8,
            num_cols=5,
            curriculum=True,
            horizontal_scale=0.12,
            vertical_scale=0.005,
            sub_terrains={
                "rough_walk": RoughWalkTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.005, 0.005),
                    noise_step=0.005,
                    downsampled_scale=0.24,
                    border_width=0.12,
                    hurdle_positions=[[1.5, 0.0], [6.5, 0.60], [11.5, -0.60]],
                    hurdle_height=0.15,
                    hurdle_height_increase=0.30,
                    hurdle_width=2.0,
                ),
            },
        )
        self.curriculum.terrain_levels = CurriculumTermCfg(
            func=RoughnessLevels,
            params={
                "promote_error": 0.15,
                "demote_error": 0.25,
                "min_base_clearance": 0.28,
                "successes_per_level": 3,
            },
        )

        self.commands.base_velocity = RoughWalkVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            heading_command=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
            ),
        )
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        # The local height grid supplies rewards. Append the waypoint after the existing proprioception.
        scanner = self.scene.height_scanner
        scanner.pattern_cfg = scanner.pattern_cfg.replace(size=(0.4, 0.3), resolution=0.1)
        self.observations.policy.height_scan = None
        self.observations.policy.hurdle_waypoint = ObservationTermCfg(func=hurdle_waypoint_observation)
        self.actions.joint_pos.clip = {".*": (-4.8, 4.8)}
        self.terminations.hurdle_course_end = TerminationTermCfg(func=hurdle_course_end, time_out=True)
        self.terminations.hurdle_waypoint_missed = TerminationTermCfg(func=hurdle_waypoint_missed)
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.8}
        )


@configclass
class UnitreeGo2RoughWalkPlayEnvCfg(UnitreeGo2RoughWalkEnvCfg):
    viewer = ViewerCfg(eye=(4.5, 8.5, 7.0), lookat=(5.0, 0.0, 0.0), origin_type="env")

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.commands.base_velocity.debug_vis = True


@configclass
class UnitreeGo2RoughWalkPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_rough_walk"
    max_iterations = 4000
    clip_actions = 4.8


def configure_two_hurdle_course(cfg):
    terrain = cfg.scene.terrain.terrain_generator.sub_terrains["rough_walk"]
    terrain.hurdle_positions = [[5.0, 0.0], [20.0, 0.0]]
    terrain.hurdle_height = 0.08
    terrain.hurdle_height_increase = 0.13
    terrain.finish_distance = 3.0

    cfg.observations.policy.hurdle_waypoint.params["nearest_hurdle"] = True
    cfg.rewards.action_smoothness.weight = -0.02
    cfg.rewards.hurdle_course_completion = RewardTermCfg(func=hurdle_course_completion_bonus, weight=2.0)


@configclass
class UnitreeGo2TwoHurdleEnvCfg(UnitreeGo2RoughWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_two_hurdle_course(self)


@configclass
class UnitreeGo2TwoHurdlePlayEnvCfg(UnitreeGo2RoughWalkPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_two_hurdle_course(self)


@configclass
class UnitreeGo2TwoHurdlePPORunnerCfg(UnitreeGo2RoughWalkPPORunnerCfg):
    preserve_walking_gait: bool = True
    max_iterations = 5000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0

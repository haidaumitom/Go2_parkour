# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import re

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--terrain", type=str, help="Generate only this terrain preset, such as parkour.")
parser.add_argument(
    "--terrain_level", type=int, help="Keep rough-walk playback on this terrain row (zero-based)."
)
parser.add_argument(
    "terrain_overrides",
    nargs="*",
    metavar="env.scene.terrain.terrain_generator.sub_terrains.NAME.FIELD=VALUE",
    help=(
        "Set terrain proportions or hurdle positions, height, depth, and width using training syntax. "
        "When proportions are specified, only terrains with listed positive weights are generated."
    ),
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

terrain_weights = {}
hurdle_settings = {}
for override in args_cli.terrain_overrides:
    match = re.fullmatch(
        r"env\.scene\.terrain\.terrain_generator\.sub_terrains\.(\w+)\."
        r"(proportion|hurdle_positions|hurdle_height|hurdle_height_increase|hurdle_depth|hurdle_width)=(.+)",
        override,
    )
    if match is None:
        parser.error(f"invalid terrain override: {override!r}")
    name, field, value = match.groups()
    if field != "proportion":
        try:
            setting = json.loads(value)
        except ValueError:
            parser.error(f"{field} must be a JSON list or number: {override!r}")
        if field == "hurdle_positions":
            if not isinstance(setting, list):
                parser.error(f"hurdle_positions must be a JSON list: {override!r}")
        elif type(setting) not in (int, float):
            parser.error(f"{field} must be a number: {override!r}")
        hurdle_settings.setdefault(name, {})[field] = setting
        continue

    try:
        weight = float(value)
    except ValueError:
        parser.error(f"terrain proportion must be a number: {override!r}")
    if not math.isfinite(weight) or weight < 0.0:
        parser.error(f"terrain proportion must be finite and non-negative: {override!r}")
    terrain_weights[name] = weight

if args_cli.terrain is not None:
    if terrain_weights:
        parser.error("choose either --terrain or terrain proportion overrides")
    terrain_weights[args_cli.terrain] = 1.0
if terrain_weights and not any(terrain_weights.values()):
    parser.error("at least one terrain must have a positive proportion")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.modules.rough_walk_runner import RoughWalkOnPolicyRunner
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from parkour_tasks.extreme_parkour_task.config.go2.rough_walk_cfg import (
    UnitreeGo2RoughWalkEnvCfg,
    UnitreeGo2RoughWalkPPORunnerCfg,
)

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def setup_free_viewport(env):
    """Frame the selected terrain with an independent camera for native viewport navigation."""
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport_and_window
    from pxr import UsdGeom

    viewport, window = get_active_viewport_and_window()
    if viewport is None or viewport.stage is None:
        return

    viewer = env.cfg.viewer
    origin = env.scene.env_origins[viewer.env_index].cpu().numpy()
    camera = UsdGeom.Camera.Define(viewport.stage, "/World/RoughWalkViewportCamera")
    camera_path = camera.GetPath()
    set_camera_view(origin + viewer.eye, origin + viewer.lookat, camera_path, viewport_api=viewport)
    viewport.camera_path = camera_path
    window.focus()


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    level = args_cli.terrain_level
    if level is not None:
        if not isinstance(env_cfg, UnitreeGo2RoughWalkEnvCfg):
            parser.error("--terrain_level is available for rough-walk tasks")
        num_rows = env_cfg.scene.terrain.terrain_generator.num_rows
        if not 0 <= level < num_rows:
            parser.error(f"--terrain_level must be between 0 and {num_rows - 1}")
        env_cfg.curriculum.terrain_levels = None

    if terrain_weights or hurdle_settings:
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is None:
            parser.error("terrain selection requires a task with a terrain generator")
        unknown = (terrain_weights.keys() | hurdle_settings.keys()) - generator.sub_terrains.keys()
        if unknown:
            available = ", ".join(generator.sub_terrains)
            parser.error(f"unknown terrains: {', '.join(sorted(unknown))}. Available terrains: {available}")
        if terrain_weights:
            generator.sub_terrains = {
                name: terrain for name, terrain in generator.sub_terrains.items() if terrain_weights.get(name, 0.0) > 0.0
            }
            for name, terrain in generator.sub_terrains.items():
                terrain.proportion = terrain_weights[name]
        for name, settings in hurdle_settings.items():
            terrain = generator.sub_terrains.get(name)
            if terrain is None or terrain.proportion <= 0.0:
                parser.error(f"hurdle overrides require an enabled terrain: {name}")
            for field, setting in settings.items():
                if not hasattr(terrain, field):
                    parser.error(f"terrain {name} does not support {field}")
                setattr(terrain, field, setting)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if level is not None:
        terrain = env.unwrapped.scene.terrain
        terrain.terrain_levels.fill_(level)
        terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        camera = env.unwrapped.viewport_camera_controller
        if camera is not None and camera.cfg.origin_type == "env":
            camera.update_view_to_env()
        print(f"[INFO]: Playback fixed at terrain level {level} (row {level + 1}/{num_rows}).")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    uses_extractor = isinstance(agent_cfg, ParkourRslRlOnPolicyRunnerCfg)
    wrapper_class = ParkourRslRlVecEnvWrapper if uses_extractor else RslRlVecEnvWrapper
    runner_class = OnPolicyRunnerWithExtractor if uses_extractor else OnPolicyRunner
    if isinstance(agent_cfg, UnitreeGo2RoughWalkPPORunnerCfg):
        runner_class = RoughWalkOnPolicyRunner
    env = wrapper_class(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    if not uses_extractor:
        policy = ppo_runner.get_inference_policy(device=env.device)
    elif agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt
    if uses_extractor:
        estimator_paras = agent_cfg.to_dict()["estimator"]
        num_prop = estimator_paras["num_prop"]
        num_scan = estimator_paras["num_scan"]
        num_priv_explicit = estimator_paras["num_priv_explicit"]
    obs, extras = env.get_observations()
    if level is not None and env_cfg.viewer.origin_type == "env" and env.unwrapped.sim.has_gui() and not args_cli.video:
        setup_free_viewport(env.unwrapped)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        if not uses_extractor:
            with torch.inference_mode():
                actions = policy(obs)
        elif agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding = True)
            # env stepping
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        obs, _, _, extras = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

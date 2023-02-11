#!/usr/bin/env python3

import argparse
import os
import os.path as osp

from habitat import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from omegaconf import OmegaConf  # keep this import for print debugging

from ovon.config import HabitatConfigPlugin


def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


def main():
    """Builds upon the habitat_baselines.run.main() function to add more flags for
    convenience."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-type",
        "-r",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Saves files to $JUNK directory and ignore resume state.",
    )
    parser.add_argument(
        "--single-env", "-s", action="store_true", help="Sets num_environments=1."
    )
    parser.add_argument(
        "--blind", "-b", action="store_true", help="If set, no cameras will be used."
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # Register custom hydra plugin
    register_plugins()

    config = get_config(args.exp_config, args.opts)
    with read_write(config):
        edit_config(config, args)

    # print(OmegaConf.to_yaml(config))
    execute_exp(config, args.run_type)


def edit_config(config, args):
    if args.debug:
        assert osp.isdir(os.environ["JUNK"]), (
            f"Environment variable directory $JUNK does not exist "
            f"(Current value: {os.environ['JUNK']})"
        )

        # Remove resume state if training
        resume_state_path = osp.join(os.environ["JUNK"], ".habitat-resume-state.pth")
        if args.run_type == "train" and osp.isfile(resume_state_path):
            print("Removing junk resume state file:", osp.abspath(resume_state_path))
            os.remove(resume_state_path)

        config.habitat_baselines.tensorboard_dir = os.environ["JUNK"]
        config.habitat_baselines.video_dir = os.environ["JUNK"]
        config.habitat_baselines.checkpoint_folder = os.environ["JUNK"]
        config.habitat_baselines.log_file = osp.join(os.environ["JUNK"], "junk.log")
        config.habitat_baselines.load_resume_state_config = False

    if args.single_env:
        config.habitat_baselines.num_environments = 1

    # Remove the frontier_exploration_map visualization from measurements if training
    if (
        args.run_type == "train"
        and "frontier_exploration_map" in config.habitat.task.measurements
    ):
        config.habitat.task.measurements.pop("frontier_exploration_map")

    # Remove all cameras if running blind (e.g., evaluating frontier explorer)
    if args.blind:
        for k in ["depth_sensor", "rgb_sensor"]:
            if k in config.habitat.simulator.agents.main_agent.sim_sensors:
                config.habitat.simulator.agents.main_agent.sim_sensors.pop(k)
        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
        )

        # A camera is required to properly load in a scene; use dummy 1x1 depth camera
        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"depth_sensor": HabitatSimDepthSensorConfig(height=1, width=1)}
        )
        if hasattr(config.habitat_baselines.rl.policy, "obs_transforms"):
            config.habitat_baselines.rl.policy.obs_transforms = {}


if __name__ == "__main__":
    main()

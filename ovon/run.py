#!/usr/bin/env python3

import argparse
import os
import os.path as osp

from habitat_baselines.run import run_exp

DEBUG_OPTIONS = {
    "habitat_baselines.tensorboard_dir": os.environ["JUNK"],
    "habitat_baselines.video_dir": os.environ["JUNK"],
    "habitat_baselines.eval_ckpt_path_dir": os.environ["JUNK"],
    "habitat_baselines.checkpoint_folder": os.environ["JUNK"],
    "habitat_baselines.log_file": osp.join(os.environ["JUNK"], "junk.log"),
}


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
        help="If set, will save files to $JUNK directory and ignore resume state.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    if args.debug:
        assert osp.isdir(os.environ["JUNK"]), (
            f"Environment variable directory $JUNK does not exist "
            f"(Current value: {os.environ['JUNK']})"
        )

        # Disable resuming
        args.opts.append("habitat_baselines.load_resume_state_config=False")

        # Override config options with DEBUG_OPTIONS
        for k, v in DEBUG_OPTIONS.items():
            args.opts.append(f"{k}={v}")

    run_exp_args = {
        k: v for k, v in vars(args).items() if k in ["run_type", "exp_config", "opts"]
    }

    run_exp(**run_exp_args)


if __name__ == "__main__":
    main()

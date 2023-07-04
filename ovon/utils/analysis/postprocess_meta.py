import argparse
from collections import defaultdict

import pandas as pd

from ovon.utils.utils import load_json, write_json


def postprocess_meta(input_path, output_path):
    df = pd.read_csv(input_path)

    categories_by_region = defaultdict(list)
    for idx, row in df.iterrows():
        categories_by_region[row["Region Proposal"]].append(row["Category"])
    
    for k, v in categories_by_region.items():
        print(k, len(v))

    write_json(categories_by_region, output_path)


def failure_metrics(input_path):
    records = load_json(input_path)

    failures = 0
    failure_modes = defaultdict(int)
    for k, v in records.items():
        failures += 1 - v["success"]
        for kk in v.keys():
            if kk in ["failure_modes.recognition", "failure_modes.exploration", "failure_modes.last_mile_nav"]:
                failure_modes[kk] += v[kk]

    print({k: v/failures for k, v in failure_modes.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, help="Path to the meta csv file", required=True
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to the output file", required=True
    )
    args = parser.parse_args()
    # postprocess_meta(args.input_path, args.output_path)
    failure_metrics(args.input_path)

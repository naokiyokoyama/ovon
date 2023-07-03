import argparse
from collections import defaultdict

import pandas as pd

from ovon.utils.utils import write_json


def postprocess_meta(input_path, output_path):
    df = pd.read_csv(input_path)

    categories_by_region = defaultdict(list)
    for idx, row in df.iterrows():
        categories_by_region[row["Region Proposal"]].append(row["Category"])
    
    for k, v in categories_by_region.items():
        print(k, len(v))

    write_json(categories_by_region, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, help="Path to the meta csv file", required=True
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to the output file", required=True
    )
    args = parser.parse_args()
    postprocess_meta(args.input_path, args.output_path)

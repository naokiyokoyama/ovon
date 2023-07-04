import argparse
import os
import random
import time
from copy import deepcopy
from math import ceil
from typing import Any, Dict, List

import openai
from tqdm import tqdm

from ovon.utils.utils import load_json, write_json

PROMPT_2D = """
{}
Based on this dictionary which contains information about 2d bounding boxes given in the form of (xmin,ymin,xmax,ymax) in a view of the target object where (0,0) is the top left cornor of the frame, write an language instruction describing the location of the target object, {}, spatially relative to other objects as references.
Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates. Add a prefix "Instruction: Find the .." or "Instruction: Go to .." to the generated instruction.
"""

PROMPT_WITH_DESCRIPTION = """
Generate an informative and natural instruction to a robot agent based on the given information(a,b):
a. Region Semantics: {}
b. Target Object Description: {}
Based on the region semantics dictionary which contains information about 2d bounding boxes given in the form of (xmin,ymin,xmax,ymax) in a view of the target object where (0,0) is the top left corner of the frame and a description of the apperance of target object write an language instruction describing the location of the target object, {}, spatially relative to other objects as references.
There are some rules:
Don't use any absolute values of the numbers, only use relative directions. Do not show bounding box coordinates in the output. Think of giving this as an instruction to a robot agent based on the given details. Add a prefix "Instruction: Find the .." or "Instruction: Go to .." to the generated instruction.
"""


PROMPT_3D = """
{}
Based on this dictionary which contains information about 3d bounding boxes of objects with center in (x,y,z) and size of the bounding box along x,y,z given in sizes_x_y_z,
write an language instruction describing the location of {} spatially relative to other objects as references. Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates.
"""

PROMPT_BOTH = """
{}
Based on this dictionary which contains information about 3d bounding boxes of objects with center in (x,y,z) and size of the bounding box along x,y,z axes given in sizes_x_y_z and 2d bounding boxes given in the form of (xmin,ymin,xmax,ymax) in this particular view,
write an language instruction describing the location of {} spatially relative to other objects as references. Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates.
"""

class PromptGenerator:
    def __init__(
        self,
        viewpoints: List[Dict],
    ) -> None:
        self.API_KEY = os.environ.get("OPENAI_APIKEY", "")
        openai.api_key = self.API_KEY
        self.viewpoints = viewpoints

    def gpt_call(
        self,
        model="gpt-3.5-turbo",
        prompt="",
    ):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            response = None
            print("Exception: {}".format(e))
        return response

    def get_prompt(self, viewpoint: Dict[str, Any], type: str = "2d", keep: float = 0):
        PROMPT = ""

        objects_in_view = viewpoint["metadata"]
        print(objects_in_view)
        target_object = [obj for obj in objects_in_view if obj["is_target"]][0]
        objects_in_view = [obj for obj in objects_in_view if not obj["is_target"]]
        objects_in_view = sorted(objects_in_view, key=lambda x: x["area"], reverse=True)

        print(target_object)

        # Drop wall from objects in view randomly
        if random.random() < 0.5:
            objects_in_view = [obj for obj in objects_in_view if not obj["category"] == "wall"]

        prompt_args = {}
        num_objects_to_sample = min(ceil(len(objects_in_view) * keep), 3)
        #objects_in_view = objects_in_view[:num_objects_to_sample]

        prompt_args[target_object["category"]] = target_object["bbox"]
        for obj in objects_in_view:
            prompt_args[obj["category"]] = obj["bbox"]

        if type == "2d":
            PROMPT = PROMPT_2D
        elif type == "with_captions":
            PROMPT = PROMPT_WITH_DESCRIPTION
        else:
            raise NotImplementedError
        
        print(viewpoint.keys())

        description = target_object["category"] + ": " + viewpoint["target_description"][0]

        prompt = PROMPT.format(prompt_args, description, target_object["category"])

        return prompt

    def generate(self, model: str = "gpt-3.5-turbo", use_openai_api: bool = True):
        drop_rate = [0.75,.25,.5]
        cnt = 0
        objects = 0
        annotated_viewpoints = {}

        for uuid, vp in tqdm(self.viewpoints.items()):
            viewpoint = deepcopy(vp)
            viewpoint["instructions"] = {}

            for d in drop_rate:
                prompt = self.get_prompt(viewpoint, keep=d, type="with_captions")
                if uuid == "picture_350":
                    print(prompt, vp["annotate_observation"])
                    print("\n\n")
                if use_openai_api:
                    response = self.gpt_call(model=model, prompt=prompt)
                    # response = None
                    if response is None:
                        viewpoint["instructions"][f"@{1-d}"] = "API_failure"
                        cnt += 1
                        print("API Call failed!")
                    else:
                        viewpoint["instructions"][f"@{1-d}"] = response["choices"][0]["message"][
                            "content"
                        ]
                else:
                    raise NotImplementedError
                break
            objects += 1
            annotated_viewpoints[uuid] = viewpoint
            time.sleep(0.2)
        print(f"API call failed for {cnt} out of {len(self.viewpoints)} objects!")
        return annotated_viewpoints

    def save_to_disk(self, data, path):
        write_json(data, path)


def main(args):
    viewpoints = load_json(args.path)

    prompt_generator = PromptGenerator(
        viewpoints=viewpoints,
    )

    viewpoint_annotations = prompt_generator.generate()
    prompt_generator.save_to_disk(viewpoint_annotations, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()
    
    main(args)

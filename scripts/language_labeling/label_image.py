import itertools
import math
import random
import time
from typing import Dict, List, Optional

import cv2
import inflect
import numpy as np
from add_text import add_text_to_image
from blip_model import BLIP2

# COLOR_PROMPT = "What color is the object in the red circle?"
COLOR_PROMPT = "Question: What color is the {}? Answer: The {} is"
# CLASS_VERIFICATION_PROMPT = "Is the object in the red circle {}?"
# CLASS_VERIFICATION_PROMPT_SINGULAR = "Is this object {}?"
CLASS_VERIFICATION_PROMPT_SINGULAR = "Question: Is this {}? Answer:"
# CLASS_VERIFICATION_PROMPT_PLURAL = "Are these {}?"
CLASS_VERIFICATION_PROMPT_PLURAL = "Question: Are these {}? Answer:"

ARRANGEMENT_PROMPT = (
    "Question: What objects are in the room? "
    "Answer: {}. "
    "Question: Where is the {} relative to the {}? "
    "Answer: The {} is"
)

LLAVA_ARRANGEMENT_PROMPT = (
    "There is {} in the picture. "
    "Describe the location of the {} relative to the other objects, using only one "
    "sentence."
)

inflect = inflect.engine()


VERBOSE = True
USE_POLLER = False
USE_COLORS = False
DRAW_RECTANGLES = False


class BLIPLabeller:
    def __init__(
        self,
        blip_model: BLIP2,
        img_rgb: np.ndarray,
        bboxes: List,
        classes: List,
        target_idx: int,
        llava_model: Optional = None,
    ):
        """

        :param blip_model:
        :param img_rgb:
        :param bboxes:
        :param classes:
        :param target_idx:
        """
        self.blip_model = blip_model
        self.img_rgb = img_rgb
        self.bboxes = bboxes
        self.classes = classes
        self.target_idx = target_idx
        self.bad_indices = []
        self.idx_to_color = {}
        self.num_objects = len(self.bboxes)
        self.llava_model = llava_model

    def label_image(self) -> List:
        self.identify_bad_objects()
        self.get_good_object_colors()
        if len(self.bad_indices) >= self.num_objects - 1:
            return []
        grounding_indices = [
            i
            for i in range(self.num_objects)
            if i not in self.bad_indices + [self.target_idx]
        ]

        # arrangements = [
        #     self.get_relative_arrangement(indices)
        #     for indices in get_combinations(grounding_indices)
        # ]

        if self.llava_model is None:
            arrangements = [
                self.get_relative_arrangement([idx])
                for idx in grounding_indices
            ]
        else:
            arrangements = [self.get_relative_arrangement(grounding_indices)]

        arrangements = [i for i in arrangements if i != ""]
        return arrangements

    def identify_bad_objects(self) -> None:
        # Check target object first
        target_is_bad = not self.object_is_recognizable(self.target_idx)

        if target_is_bad:
            self.bad_indices = list(range(self.num_objects))
        else:
            for idx in range(self.num_objects):
                if idx != self.target_idx and not self.object_is_recognizable(
                    idx
                ):
                    self.bad_indices.append(idx)

    def get_good_object_colors(self) -> None:
        if not USE_COLORS:
            return
        for idx in range(self.num_objects):
            if idx not in self.bad_indices:
                self.idx_to_color[idx] = self.get_color(idx)

    def object_is_recognizable(self, idx: int) -> bool:
        image_chip = self.crop_with_padding(idx)
        class_name = self.classes[idx].lower()
        if is_plural(class_name):
            class_name = add_a_or_an(class_name)
            prompt = CLASS_VERIFICATION_PROMPT_SINGULAR
        else:
            prompt = CLASS_VERIFICATION_PROMPT_PLURAL
        prompt = prompt.format(class_name)
        blip_response = self.ask_blip(image_chip, prompt)
        return (
            blip_response.lower().startswith("yes")
            or f"it is {add_a_or_an(class_name)}" in blip_response.lower()
            or blip_response.lower().startswith(add_a_or_an(class_name))
        )

    def get_relative_arrangement(self, indices) -> str:
        """
        :param target_class: class to get relative arrangement for
        :return: relative arrangement of target_class to that object
        """
        unique_classes = set()
        grounding_objects = []
        target_class = None
        for idx in range(self.num_objects):
            if idx in self.bad_indices:
                continue
            class_name = f"{self.classes[idx]}"
            if class_name in ["picture", "photo"]:
                class_name = "decorative " + class_name
            if USE_COLORS:
                class_name = f"{self.idx_to_color[idx]} {class_name}"
            unique_classes.add(class_name)
            if idx == self.target_idx or idx in indices:
                grounding_objects.append((class_name, self.bboxes[idx]))
                if idx == self.target_idx:
                    target_class = class_name

        assert target_class is not None

        grounding_objects = remove_duplicates(grounding_objects)
        if DRAW_RECTANGLES:
            blip_img = self.img_rgb.copy()
            for _, bbox in grounding_objects:
                x, y, w, h = get_padded_bbox(blip_img, bbox)
                top_left_pt = (x, y)
                bottom_right_pt = (x + w, y + h)
                cv2.rectangle(
                    blip_img, top_left_pt, bottom_right_pt, (255, 0, 0), 4
                )
        else:
            blip_img = self.img_rgb
        grounding_objects = set([i[0] for i in grounding_objects])
        if target_class in grounding_objects:
            grounding_objects.remove(target_class)

        if len(unique_classes) < 2:
            return ""
        all_classes_str = join_words([add_a_or_an(i) for i in unique_classes])

        grounding_objects = list(grounding_objects)
        if len(grounding_objects) == 0:
            return ""
        elif len(grounding_objects) == 1:
            grounding_objects = grounding_objects[0]
        else:
            grounding_objects = join_words(
                ["the " + i for i in grounding_objects]
            )

        if self.llava_model is None:
            prompt = ARRANGEMENT_PROMPT.format(
                all_classes_str, target_class, grounding_objects, target_class
            )
            relative_arrangement = self.ask_blip(blip_img, prompt)
        else:
            prompt = LLAVA_ARRANGEMENT_PROMPT.format(
                all_classes_str, target_class
            )
            relative_arrangement = self.llava_model.eval(blip_img, prompt)
            print(prompt)
            print(relative_arrangement)

        return target_class + " " + relative_arrangement

    # def verify_class(self, idx: int) -> str:
    #     circled_img = self.draw_red_ellipse_idx(idx)
    #     class_verification = self.ask_blip(
    #         circled_img,
    #         CLASS_VERIFICATION_PROMPT.format(add_a_or_an(self.classes[idx])),
    #     )
    #     return class_verification

    def get_color(self, idx: int) -> str:
        # circled_img = self.draw_red_ellipse_idx(idx)
        # color = self.ask_blip(circled_img, COLOR_PROMPT)
        # return color
        image_chip = self.crop_with_padding(idx)
        class_name = self.classes[idx]
        color = self.ask_blip(
            image_chip, COLOR_PROMPT.format(class_name, class_name)
        )
        return color

    def ask_blip(self, img_rgb, prompt=None):
        if USE_POLLER:
            import os.path as osp

            dirname = "/coc/testnvme/nyokoyama3/summer_2023/language_nav_labeler/for_blip"
            filename = osp.join(dirname, str(time.time()) + ".png")
            with open(filename.replace(".png", ".txt"), "w") as f:
                f.write("" if prompt is None else prompt)
            cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            response_name = filename[:-3] + "response"
            while not osp.isfile(response_name):
                time.sleep(0.05)
            with open(response_name) as f:
                answer = f.read()
        else:
            answer = self.blip_model.ask(img_rgb, prompt)[0]

        if VERBOSE:
            print("Prompt:")
            print(prompt)
            print("Answer:")
            print(answer)

            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img = add_text_to_image(img, "Prompt: " + prompt)
            img = add_text_to_image(img, "Answer: " + answer)
            filename = "visualizations/" + str(time.time()) + ".jpg"
            cv2.imwrite(filename, img)

        return answer
        # import os.path as osp
        # import time
        #
        # dirname = "/coc/testnvme/nyokoyama3/summer_2023/language_nav_labeler/for_blip"
        # filename = osp.join(dirname, str(time.time()) + ".png")
        # with open(filename.replace(".png", ".txt"), "w") as f:
        #     f.write("" if prompt is None else prompt)
        # cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        # return "test"

    def draw_red_ellipse_idx(self, idx: int) -> np.ndarray:
        pt1, pt2 = self.bboxes[idx]
        circled_img = draw_red_ellipse(self.img_rgb, pt1, pt2)
        return circled_img

    def crop_with_padding(self, idx, padding=15):
        image = self.img_rgb
        bounding_box = self.bboxes[idx]
        x, y, w, h = get_padded_bbox(image, bounding_box, padding)

        # Crop the image with padding
        cropped_image = image[y : y + h, x : x + w]

        return cropped_image


def get_padded_bbox(image, bounding_box, padding=15):
    # Extract image dimensions
    image_height, image_width = image.shape[:2]

    # Extract bounding box coordinates
    x1, y1 = bounding_box[0]
    x2, y2 = bounding_box[1]

    # Calculate the new bounding box coordinates with padding
    x = max(0, min(x1, x2) - padding)
    y = max(0, min(y1, y2) - padding)
    w = min(abs(x1 - x2) + 2 * padding, image_width - x)
    h = min(abs(y1 - y2) + 2 * padding, image_height - y)

    return x, y, w, h


def draw_red_ellipse(img_rgb, pt1, pt2):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2

    # Calculate the radius of the ellipse
    dx = abs(pt2[0] - pt1[0]) // 2
    dy = abs(pt2[1] - pt1[1]) // 2

    radius_x = int(math.sqrt(dx * (dx + dy)))
    radius_y = int(math.sqrt(dy * (dx + dy)))

    img_bgr = cv2.ellipse(
        img_bgr,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        360,
        (0, 0, 255),
        2,
    )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def add_a_or_an(word):
    if is_plural(word):
        prefix = "an " if starts_with_vowel(word) else "a "
        return prefix + word
    return word


def starts_with_vowel(word):
    vowels = ["a", "e", "i", "o", "u"]
    first_letter = word.lower()[0]
    return first_letter in vowels


def parse_json(json_path):
    import json

    with open(json_path, "r") as file:
        data = json.load(file)

    bboxes, classes = [], []
    for obj in data["results"]:
        bboxes.append((obj["top_left"], obj["bottom_right"]))
        classes.append(obj["class"])

    return bboxes, classes


def join_words(words):
    if len(words) <= 2:
        return " and ".join(words)
    else:
        return ", ".join(words[:-1]) + ", and " + words[-1]


def get_combinations(nums: List):
    combinations = set()
    for r in range(1, len(nums) + 1):
        combinations.update(
            tuple(set(comb)) for comb in itertools.combinations(nums, r)
        )
    return combinations


def is_plural(word: str) -> bool:
    return not inflect.singular_noun(word.split()[-1])


def remove_duplicates(tuples_list: List) -> List:
    shuffled_list = list(tuples_list)
    random.shuffle(shuffled_list)

    unique_strings = set()
    result = []

    for item in shuffled_list:
        string = item[0]
        if string not in unique_strings:
            result.append(item)
            unique_strings.add(string)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze image with BLIP-2.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("json_path", help="Path to the JSON file")
    args = parser.parse_args()

    img_rgb = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_RGB2BGR)
    bboxes, classes = parse_json(args.json_path)

    # model = BLIP2()
    b = BLIPLabeller(
        blip_model=None,
        img_rgb=img_rgb,
        bboxes=bboxes,
        classes=classes,
        target_idx=0,
    )
    b.label_image()

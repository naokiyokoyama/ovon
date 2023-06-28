import argparse
import os
from io import BytesIO
from typing import Union

import numpy as np
import requests
import torch
from llava.conversation import SeparatorStyle, conv_templates
from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        if image.startswith("http") or image.startswith("https"):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        raise ValueError("Unsupported image file format")

    return image


class LLaVA:
    def __init__(self, model_name):
        # Model
        disable_torch_init()
        model_name = os.path.expanduser(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "mpt" in model_name.lower():
            model = LlavaMPTForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_cache=True,
            ).cuda()
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_cache=True,
            ).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

        mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False
        )
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )

        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == "meta":
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device="cuda", dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.use_im_start_end = mm_use_im_start_end

        self.mm_use_im_start_end = mm_use_im_start_end
        self.vision_config = vision_config
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.image_processor = image_processor
        self.model = model

    def eval(self, image_file: Union[str, np.ndarray], query: str):
        mm_use_im_start_end = self.mm_use_im_start_end
        vision_config = self.vision_config
        tokenizer = self.tokenizer
        model_name = self.model_name
        image_processor = self.image_processor
        model = self.model

        if mm_use_im_start_end:
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )
        image_token_len = (
            vision_config.image_size // vision_config.patch_size
        ) ** 2

        qs = query
        if mm_use_im_start_end:
            qs = (
                qs
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + DEFAULT_IM_END_TOKEN
            )
        else:
            qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt_multimodal"
        else:
            conv_mode = "multimodal"

        if conv_mode is not None and conv_mode != conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` "
                "is {}, using {}".format(conv_mode, conv_mode, conv_mode)
            )
        else:
            conv_mode = conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        image = load_image(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = (
            conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the "
                f"input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs


class LLaVALabeller:
    def __init__(self, model: LLaVA, img_rgb, bboxes, classes, target_idx):
        self.model = model
        self.img_rgb = img_rgb
        self.bboxes = bboxes
        self.classes = classes
        self.target_idx = target_idx

    def label_image(self):
        target_object = self.classes[self.target_idx]
        query = (
            f"Describe where the {target_object} is relative to the other objects, "
            "using only one sentence."
        )
        answer = self.model.eval(query, self.img_rgb)
        print(query)
        print(answer)
        return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    l = LLaVA(args.model_name)
    print(l.eval(args.query, args.image_file))

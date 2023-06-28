import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class BLIP2:
    def __init__(self, device=None):
        if device is None:
            # setup device to use
            device = (
                torch.device("cuda") if torch.cuda.is_available() else "cpu"
            )

        # loads BLIP-2 pre-trained model
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )
        self.device = device

    def ask(self, image, prompt=None, processed_image=None):
        if processed_image is None:
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            else:
                pil_img = image

            processed_image = (
                self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
            )

        if prompt is None:
            out = self.model.generate({"image": processed_image})
        else:
            out = self.model.generate(
                {"image": processed_image, "prompt": prompt}
            )

        return out

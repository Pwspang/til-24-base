from typing import List
from transformers import pipeline
from PIL import Image
from io import BytesIO
from torchvision.transforms import v2 as T
import torch
import numpy as np
from inference import load_model, get_grounding_output, load_image


class VLMManager:
    def __init__(self):
        # initialize the model here
        checkpoint = "checkpoint_best_regular.pth"
        config = "GroundingDINO_SwinT_OGC.py"
        
        self.model = load_model(config, checkpoint)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        image_pil, image = load_image(Image.open(BytesIO(image)).convert("RGB"))
        W, H = image_pil.size
        
        bboxes, captions = get_grounding_output(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=0.15,
            text_threshold=0.25
        )
        
        if len(bboxes) > 0:
            box = bboxes[0] * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            return [x0, y0, x1-x0, y1-y0]
        
        try:
            bboxes, captions = get_grounding_output(
                model=self.model,
                image=image,
                caption=caption.split(" ")[-1],
                box_threshold=0.15,
                text_threshold=0.25
            )
            box = bboxes[0] * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            return [x0, y0, x1-x0, y1-y0]
        except:
            return [0, 0, 0, 0]


if __name__ == "__main__":
    vlm = VLMManager() 
    with open("/home/jupyter/advanced/images/image_0.jpg", "rb") as file:
        image_bytes = file.read()
        print(vlm.identify(image_bytes, "grey missile"))

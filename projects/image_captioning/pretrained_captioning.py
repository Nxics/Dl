"""Optional pretrained image-captioning helpers for the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


BLIP_MODEL_NAME = 'Salesforce/blip-image-captioning-base'


@dataclass
class PretrainedCaptioner:
    """Small wrapper around a pretrained Hugging Face captioning model."""

    processor: object
    model: object
    device: str

    def generate(self, image: Image.Image, max_length: int = 30) -> str:
        inputs = self.processor(
            images=image.convert('RGB'),
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
            )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)


def load_blip_captioner(device: str = 'cpu',
                        model_name: str = BLIP_MODEL_NAME) -> PretrainedCaptioner:
    """Load a pretrained BLIP captioning model.

    The import is intentionally inside the function so the core project remains
    runnable even when optional Hugging Face dependencies are not installed.
    """

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'Optional dependency `transformers` is required for pretrained '
            'captioning. Install it with `python -m pip install -r requirements.txt`.'
        ) from error

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return PretrainedCaptioner(processor=processor, model=model, device=device)

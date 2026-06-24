import torch
from PIL import Image

from projects.image_captioning.pretrained_captioning import PretrainedCaptioner


class FakeInputs(dict):

    def to(self, device: str):
        self['device'] = device
        return self


class FakeProcessor:

    def __call__(self, images, return_tensors: str):
        assert images.mode == 'RGB'
        assert return_tensors == 'pt'
        return FakeInputs(pixel_values=torch.zeros(1, 3, 2, 2))

    def decode(self, output_ids, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return 'a dog is sitting in the grass'


class FakeModel:

    def generate(self, **inputs):
        assert 'pixel_values' in inputs
        return torch.tensor([[1, 2, 3]])


def test_when_pretrained_captioner_generates_then_returns_decoded_text():
    captioner = PretrainedCaptioner(
        processor=FakeProcessor(),
        model=FakeModel(),
        device='cpu',
    )
    image = Image.new('RGB', (8, 8), color='white')

    caption = captioner.generate(image)

    assert caption == 'a dog is sitting in the grass'

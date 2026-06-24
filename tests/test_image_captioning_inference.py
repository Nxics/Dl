import torch

from projects.image_captioning.inference import _beam_search_decode
from projects.image_captioning.vocab import Vocabulary


class TinyDecoder(torch.nn.Module):

    def __init__(self, vocab_size: int, end_idx: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.end_idx = end_idx

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = captions.shape
        logits = torch.full((batch_size, sequence_length, self.vocab_size), -10.0)
        logits[:, -1, 4] = 5.0
        logits[:, -1, self.end_idx] = 4.0
        return logits


class TinyCaptionModel(torch.nn.Module):

    def __init__(self, vocab_size: int, end_idx: int) -> None:
        super().__init__()
        self.decoder = TinyDecoder(vocab_size, end_idx)


def test_when_beam_search_runs_then_returns_decoded_caption_without_special_tokens():
    vocabulary = Vocabulary()
    vocabulary.fit(['dog runs'])
    model = TinyCaptionModel(len(vocabulary), vocabulary.end_idx)
    image_features = torch.randn(1, 8)

    caption = _beam_search_decode(
        model,
        image_features,
        vocabulary,
        max_length=3,
        device='cpu',
        beam_size=2,
    )

    assert caption
    assert '<start>' not in caption
    assert '<end>' not in caption

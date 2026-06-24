import torch
from PIL import Image

from projects.image_captioning.retrieval import (
    VggRetrievalCaptioner,
    normalize_features,
    pool_backbone_feature,
)


class FakeEncoder(torch.nn.Module):

    def __init__(self, raw_feature: torch.Tensor) -> None:
        super().__init__()
        self.raw_feature = raw_feature

    def extract_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        del images
        return self.raw_feature.unsqueeze(0)


def fake_transform(image: Image.Image) -> torch.Tensor:
    del image
    return torch.zeros(3, 224, 224)


def make_raw_feature(compact_feature: torch.Tensor) -> torch.Tensor:
    return compact_feature.repeat_interleave(49)


def test_when_backbone_feature_pooled_then_returns_512_dimensions():
    compact = torch.arange(512, dtype=torch.float32)
    raw = make_raw_feature(compact)

    pooled = pool_backbone_feature(raw)

    assert tuple(pooled.shape) == (512,)
    assert torch.allclose(pooled, compact)


def test_when_retrieval_runs_then_nearest_caption_is_returned():
    dog_feature = torch.zeros(512)
    dog_feature[0] = 1.0
    bench_feature = torch.zeros(512)
    bench_feature[1] = 1.0
    query_raw_feature = make_raw_feature(dog_feature)

    captioner = VggRetrievalCaptioner(
        image_names=['dog.jpg', 'bench.jpg'],
        captions_by_image={
            'dog.jpg': ['a dog is sitting in the grass'],
            'bench.jpg': ['a woman is sitting on a bench'],
        },
        feature_matrix=torch.stack([dog_feature, bench_feature]),
        encoder=FakeEncoder(query_raw_feature),
        transform=fake_transform,
        device='cpu',
    )

    caption, matches = captioner.generate(Image.new('RGB', (8, 8)), top_k=2)

    assert caption == 'a dog is sitting in the grass'
    assert matches[0].image == 'dog.jpg'
    assert matches[0].similarity > matches[1].similarity


def test_when_features_normalized_then_rows_have_unit_norm():
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    normalized = normalize_features(features)

    assert torch.allclose(torch.linalg.norm(normalized, dim=1), torch.ones(2))

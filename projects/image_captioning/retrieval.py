"""VGG feature retrieval baseline for more visually grounded captions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.torch'))

from projects.image_captioning.model import VggEncoder


def pool_backbone_feature(feature: torch.Tensor) -> torch.Tensor:
    """Convert a VGG pool7 feature to a compact 512-dimensional vector."""

    feature = feature.float()
    if feature.ndim == 2:
        return feature.reshape(feature.shape[0], 512, 7, 7).mean(dim=(2, 3))
    return feature.reshape(512, 7, 7).mean(dim=(1, 2))


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalize features for cosine-similarity retrieval."""

    return torch.nn.functional.normalize(features.float(), dim=-1)


@dataclass(frozen=True)
class RetrievalMatch:
    image: str
    caption: str
    similarity: float


class VggRetrievalCaptioner:
    """Retrieve captions from visually similar Flickr8k training images."""

    def __init__(self,
                 image_names: list[str],
                 captions_by_image: dict[str, list[str]],
                 feature_matrix: torch.Tensor,
                 encoder: VggEncoder,
                 transform,
                 device: torch.device | str = 'cpu') -> None:
        self.image_names = image_names
        self.captions_by_image = captions_by_image
        self.feature_matrix = normalize_features(feature_matrix).cpu()
        self.encoder = encoder.eval().to(device)
        self.transform = transform
        self.device = torch.device(device)

    @classmethod
    def from_cached_features(cls,
                             captions_frame: pd.DataFrame,
                             features_dir: str | Path,
                             transform,
                             device: torch.device | str = 'cpu') -> 'VggRetrievalCaptioner':
        captions_by_image = (
            captions_frame
            .groupby('image')['caption']
            .apply(list)
            .to_dict()
        )
        image_names: list[str] = []
        pooled_features: list[torch.Tensor] = []

        for image_name in sorted(captions_by_image):
            feature_path = Path(features_dir) / f'{Path(image_name).stem}.pt'
            if not feature_path.is_file():
                continue
            feature = torch.load(feature_path, map_location='cpu', weights_only=True)
            image_names.append(image_name)
            pooled_features.append(pool_backbone_feature(feature))

        if not pooled_features:
            raise ValueError(f'No cached VGG features found in {features_dir}')

        feature_matrix = torch.stack(pooled_features)
        encoder = VggEncoder(pretrained=True, freeze=True)
        return cls(
            image_names=image_names,
            captions_by_image=captions_by_image,
            feature_matrix=feature_matrix,
            encoder=encoder,
            transform=transform,
            device=device,
        )

    def retrieve(self, image: Image.Image, top_k: int = 5) -> list[RetrievalMatch]:
        """Return captions from the top-k visually closest training images."""

        image_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            backbone_feature = self.encoder.extract_backbone_features(image_tensor).cpu()
        query_feature = normalize_features(pool_backbone_feature(backbone_feature))
        similarities = torch.matmul(self.feature_matrix, query_feature.squeeze(0))
        top_scores, top_indices = torch.topk(
            similarities,
            k=min(top_k, len(self.image_names)),
        )

        matches: list[RetrievalMatch] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            image_name = self.image_names[int(index)]
            captions = self.captions_by_image[image_name]
            caption = min(captions, key=lambda text: (abs(len(text.split()) - 10), len(text)))
            matches.append(
                RetrievalMatch(
                    image=image_name,
                    caption=caption,
                    similarity=float(score),
                )
            )
        return matches

    def generate(self, image: Image.Image, top_k: int = 5) -> tuple[str, list[RetrievalMatch]]:
        matches = self.retrieve(image, top_k=top_k)
        return matches[0].caption if matches else '', matches

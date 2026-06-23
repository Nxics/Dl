from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from projects.image_captioning.vocab import Vocabulary


def read_flickr8k_captions(captions_path: str | Path) -> pd.DataFrame:
    """Read Flickr8k captions from common Kaggle formats.

    Supported rows look like either:
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress...

    or Kaggle pipe-separated rows:
    image_name|caption_number|caption_text
    1000268201_693b08cb0e.jpg|0|A child in a pink dress...

    or legacy rows like:
    1000268201_693b08cb0e.jpg#0\tA child in a pink dress...
    """
    captions_path = Path(captions_path)
    if not captions_path.exists():
        raise FileNotFoundError(f'Caption file not found: {captions_path}')

    pipe_frame = pd.read_csv(captions_path, sep='|')
    if {'image_name', 'caption_text'}.issubset(pipe_frame.columns):
        return (
            pipe_frame[['image_name', 'caption_text']]
            .rename(columns={'image_name': 'image', 'caption_text': 'caption'})
            .dropna()
            .reset_index(drop=True)
        )

    try:
        frame = pd.read_csv(captions_path)
        if {'image', 'caption'}.issubset(frame.columns):
            return frame[['image', 'caption']].dropna().reset_index(drop=True)
    except pd.errors.ParserError:
        pass

    rows: list[dict[str, str]] = []
    with captions_path.open(encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                image_part, caption = line.split('\t', maxsplit=1)
            elif line.count('|') >= 2:
                image_part, _, caption = line.split('|', maxsplit=2)
                if image_part == 'image_name':
                    continue
            elif ',' in line:
                image_part, caption = line.split(',', maxsplit=1)
            else:
                continue
            image_name = image_part.split('#')[0]
            rows.append({'image': image_name, 'caption': caption})
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


class Flickr8kCaptionDataset(Dataset):
    """Dataset returning one image tensor and one encoded caption."""

    def __init__(self,
                 images_dir: str | Path,
                 captions_frame: pd.DataFrame,
                 vocabulary: Vocabulary,
                 transform: Callable | None = None,
                 max_length: int | None = None) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.captions_frame = captions_frame.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_length = max_length
        if self.max_length is not None and self.max_length < 2:
            raise ValueError('max_length must allow at least <start> and <end> tokens')

    def __len__(self) -> int:
        return len(self.captions_frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.captions_frame.iloc[index]
        image_path = self.images_dir / row['image']
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            from torchvision import transforms
            image_tensor = transforms.ToTensor()(image)

        caption_indices = self.vocabulary.encode(row['caption'])
        if self.max_length is not None:
            if len(caption_indices) > self.max_length:
                caption_indices = (
                    caption_indices[:self.max_length - 1]
                    + [self.vocabulary.end_idx]
                )
            padding = [self.vocabulary.pad_idx] * (self.max_length - len(caption_indices))
            caption_indices = caption_indices + padding

        return image_tensor, torch.tensor(caption_indices, dtype=torch.long)


class Flickr8kFeatureCaptionDataset(Flickr8kCaptionDataset):
    """Dataset returning cached VGG features and one encoded caption."""

    def __init__(self,
                 features_dir: str | Path,
                 captions_frame: pd.DataFrame,
                 vocabulary: Vocabulary,
                 max_length: int | None = None) -> None:
        super().__init__(
            images_dir=features_dir,
            captions_frame=captions_frame,
            vocabulary=vocabulary,
            max_length=max_length,
        )
        self.features_dir = Path(features_dir)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.captions_frame.iloc[index]
        feature_path = self.features_dir / f"{Path(row['image']).stem}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(f'Cached feature not found: {feature_path}')

        features = torch.load(
            feature_path,
            map_location='cpu',
            weights_only=True,
        ).float()
        caption_indices = self.vocabulary.encode(row['caption'])
        if self.max_length is not None:
            if len(caption_indices) > self.max_length:
                caption_indices = (
                    caption_indices[:self.max_length - 1]
                    + [self.vocabulary.end_idx]
                )
            caption_indices += [self.vocabulary.pad_idx] * (
                self.max_length - len(caption_indices)
            )

        return features, torch.tensor(caption_indices, dtype=torch.long)


def make_caption_collate_fn(pad_idx: int):
    """Create a collate function that pads captions in a batch."""

    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        images, captions = zip(*batch)
        images_tensor = torch.stack(list(images))
        captions_tensor = torch.nn.utils.rnn.pad_sequence(list(captions),
                                                          batch_first=True,
                                                          padding_value=pad_idx)
        return images_tensor, captions_tensor

    return collate_fn

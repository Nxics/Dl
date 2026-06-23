from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _UniqueImageDataset(Dataset):

    def __init__(self, images_dir: str | Path, image_names: list[str], transform) -> None:
        self.images_dir = Path(images_dir)
        self.image_names = image_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_name = self.image_names[index]
        with Image.open(self.images_dir / image_name) as image:
            image_tensor = self.transform(image.convert('RGB'))
        return image_tensor, image_name


def cache_vgg_backbone_features(encoder,
                                images_dir: str | Path,
                                captions_frame: pd.DataFrame,
                                output_dir: str | Path,
                                transform,
                                device: torch.device | str = 'cpu',
                                batch_size: int = 32,
                                num_workers: int = 0,
                                overwrite: bool = False,
                                max_images: int | None = None) -> dict[str, int]:
    """Extract and save one pooled VGG feature tensor per unique image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted(captions_frame['image'].unique())
    if max_images is not None:
        image_names = image_names[:max_images]

    pending_names = [
        name for name in image_names
        if overwrite or not (output_dir / f'{Path(name).stem}.pt').is_file()
    ]
    if not pending_names:
        return {'requested': len(image_names), 'created': 0, 'existing': len(image_names)}

    loader = DataLoader(
        _UniqueImageDataset(images_dir, pending_names, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    encoder.eval()
    encoder.to(device)
    created = 0
    with torch.inference_mode():
        for images, names in loader:
            features = encoder.extract_backbone_features(images.to(device))
            features = features.detach().cpu().to(torch.float16)
            for feature, image_name in zip(features, names):
                # Clone detaches the slice from the full batch storage. Without
                # it, every small feature file serializes the whole batch.
                torch.save(
                    feature.clone(),
                    output_dir / f'{Path(image_name).stem}.pt',
                )
                created += 1

    return {
        'requested': len(image_names),
        'created': created,
        'existing': len(image_names) - created,
    }

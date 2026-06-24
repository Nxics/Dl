"""Check common setup issues for the image-captioning project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.image_captioning.data import read_flickr8k_captions
from projects.image_captioning.transforms import describe_image_preprocessing


def diagnose_setup(data_dir: Path, checkpoint_path: Path) -> list[str]:
    messages: list[str] = []

    captions_path = data_dir / 'captions.txt'
    images_dir = data_dir / 'Images'

    if captions_path.exists():
        captions = read_flickr8k_captions(captions_path)
        messages.append(f'captions: {len(captions):,} rows')
        messages.append(f'unique images in captions: {captions["image"].nunique():,}')
    else:
        messages.append(f'missing captions file: {captions_path}')

    if images_dir.exists():
        image_count = sum(1 for path in images_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        messages.append(f'image files: {image_count:,}')
    else:
        messages.append(f'missing images directory: {images_dir}')

    preprocessing = describe_image_preprocessing()
    messages.append(
        'preprocessing: '
        f'resize={preprocessing["resize"]}, '
        f'mean={preprocessing["normalization_mean"]}, '
        f'std={preprocessing["normalization_std"]}'
    )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        messages.append(f'checkpoint: {checkpoint_path}')
        messages.append(f'checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
        messages.append(f'validation loss: {checkpoint.get("validation_loss", "unknown")}')
        messages.append(f'vocabulary size: {len(checkpoint.get("vocabulary", {}).get("token_to_idx", {})):,}')
        if checkpoint.get('preprocessing_config') is None:
            messages.append('checkpoint preprocessing metadata: missing; use current project preprocessing')
        else:
            messages.append('checkpoint preprocessing metadata: present')
    else:
        messages.append(f'missing checkpoint: {checkpoint_path}')

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description='Diagnose Flickr8k captioning setup.')
    parser.add_argument('--data-dir', type=Path, default=Path('DATA/flickr8k'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best_model.pt'))
    args = parser.parse_args()

    for message in diagnose_setup(args.data_dir, args.checkpoint):
        print(message)


if __name__ == '__main__':
    main()

from pathlib import Path

import torch
from PIL import Image

from scripts.diagnose_captioning_setup import diagnose_setup


def test_when_diagnose_setup_runs_then_reports_dataset_and_checkpoint(tmp_path):
    data_dir = tmp_path / 'flickr8k'
    images_dir = data_dir / 'Images'
    images_dir.mkdir(parents=True)
    Image.new('RGB', (4, 4), color='white').save(images_dir / 'img.jpg')
    (data_dir / 'captions.txt').write_text(
        'image_name|caption_number|caption_text\n'
        'img.jpg|0|a dog runs\n',
        encoding='utf-8',
    )
    checkpoint_path = tmp_path / 'best_model.pt'
    torch.save(
        {
            'epoch': 3,
            'validation_loss': 2.5,
            'vocabulary': {'token_to_idx': {'<pad>': 0, 'dog': 1}},
        },
        checkpoint_path,
    )

    messages = diagnose_setup(data_dir, checkpoint_path)

    assert any('captions:' in message for message in messages)
    assert any('checkpoint epoch: 3' in message for message in messages)

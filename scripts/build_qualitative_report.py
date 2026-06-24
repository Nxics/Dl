"""Build a qualitative comparison report for generated captions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.image_captioning.evaluation import load_caption_checkpoint  # noqa: E402
from projects.image_captioning.inference import generate_caption_from_backbone_features  # noqa: E402
from projects.image_captioning.retrieval import VggRetrievalCaptioner  # noqa: E402
from projects.image_captioning.transforms import get_eval_transforms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare LSTM and VGG retrieval captions qualitatively.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'DATA' / 'flickr8k')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'best_model.pt')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports' / 'model' / 'qualitative_comparison.csv')
    parser.add_argument('--examples', type=int, default=12)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    splits_dir = args.data_dir / 'splits'
    features_dir = args.data_dir / 'features' / 'vgg16_pool7'
    train_captions = pd.read_csv(splits_dir / 'train.csv')
    test_captions = pd.read_csv(splits_dir / 'test.csv')
    model, vocabulary, checkpoint = load_caption_checkpoint(args.checkpoint, args.device)
    retrieval_captioner = VggRetrievalCaptioner.from_cached_features(
        train_captions,
        features_dir,
        get_eval_transforms(),
        device=args.device,
    )

    grouped = test_captions.groupby('image')['caption'].apply(list)
    selected_images = list(grouped.index[:args.examples])
    rows = []
    for image_name in selected_images:
        references = grouped[image_name]
        feature_path = features_dir / f'{Path(image_name).stem}.pt'
        features = torch.load(feature_path, map_location='cpu', weights_only=True)
        lstm_caption = generate_caption_from_backbone_features(
            model,
            features,
            vocabulary,
            max_length=checkpoint.get('max_caption_length', 20),
            device=args.device,
            decoding='beam',
            beam_size=3,
        )
        image = Image.open(args.data_dir / 'Images' / image_name).convert('RGB')
        retrieval_caption, retrieval_matches = retrieval_captioner.generate(image, top_k=3)
        rows.append({
            'image': image_name,
            'references': ' | '.join(references),
            'vgg_lstm_caption': lstm_caption,
            'vgg_retrieval_caption': retrieval_caption,
            'retrieval_top_match': retrieval_matches[0].image if retrieval_matches else '',
            'retrieval_similarity': retrieval_matches[0].similarity if retrieval_matches else 0.0,
        })

    output_frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False)
    print(output_frame.to_string(index=False))
    print(f'Saved qualitative report to {args.output}')


if __name__ == '__main__':
    main()

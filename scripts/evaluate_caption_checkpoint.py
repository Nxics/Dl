"""Evaluate a captioning checkpoint on the Flickr8k test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.image_captioning.data import (  # noqa: E402
    Flickr8kFeatureCaptionDataset,
    make_caption_collate_fn,
)
from projects.image_captioning.evaluation import (  # noqa: E402
    corpus_bleu,
    load_caption_checkpoint,
    rouge_l_f1,
)
from projects.image_captioning.inference import generate_caption_from_backbone_features  # noqa: E402
from projects.image_captioning.training import run_caption_epoch  # noqa: E402
from projects.image_captioning.vocab import tokenize  # noqa: E402


def select_device(preferred: str) -> torch.device:
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    device = select_device(args.device)
    model, vocabulary, checkpoint = load_caption_checkpoint(args.checkpoint, device)

    test_captions = pd.read_csv(args.data_dir / 'splits' / 'test.csv')
    features_dir = args.data_dir / 'features' / 'vgg16_pool7'
    collate_fn = make_caption_collate_fn(vocabulary.pad_idx)
    test_dataset = Flickr8kFeatureCaptionDataset(
        features_dir,
        test_captions,
        vocabulary,
        checkpoint.get('max_caption_length', args.max_caption_length),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
    test_loss = run_caption_epoch(model, test_loader, criterion, device)

    generated_rows: list[dict[str, str]] = []
    metric_references: list[list[list[str]]] = []
    metric_hypotheses: list[list[str]] = []

    grouped = test_captions.groupby('image')['caption'].apply(list)
    for image_index, (image_name, references) in enumerate(grouped.items(), start=1):
        feature_path = features_dir / f'{Path(image_name).stem}.pt'
        features = torch.load(feature_path, map_location='cpu', weights_only=True)
        caption = generate_caption_from_backbone_features(
            model,
            features,
            vocabulary,
            max_length=checkpoint.get('max_caption_length', args.max_caption_length),
            device=device,
            decoding=args.decoding,
            beam_size=args.beam_size,
        )
        generated_rows.append({
            'image': image_name,
            'generated_caption': caption,
            'references': ' | '.join(references),
        })
        metric_references.append([tokenize(reference) for reference in references])
        metric_hypotheses.append(tokenize(caption))

        if args.log_every and image_index % args.log_every == 0:
            print(f'generated {image_index:,}/{len(grouped):,} captions', flush=True)

    metrics = {
        'evaluated_images': float(len(generated_rows)),
        'BLEU-1': corpus_bleu(metric_references, metric_hypotheses, max_order=1),
        'BLEU-4': corpus_bleu(metric_references, metric_hypotheses, max_order=4),
        'ROUGE-L': rouge_l_f1(metric_references, metric_hypotheses),
        'test_loss': test_loss,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generated_rows).to_csv(
        args.output_dir / 'generated_captions.csv',
        index=False,
    )
    pd.Series(metrics).to_csv(args.output_dir / 'evaluation_metrics.csv')

    print(pd.Series(metrics).to_string(), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Flickr8k captioning checkpoint.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'DATA' / 'flickr8k')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'best_model.pt')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'reports' / 'model')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-caption-length', type=int, default=20)
    parser.add_argument('--decoding', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam-size', type=int, default=3)
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()

    evaluate_checkpoint(args)


if __name__ == '__main__':
    main()

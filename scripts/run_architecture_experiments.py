"""Train small VGG+LSTM architecture variants on cached Flickr8k features."""

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
from projects.image_captioning.evaluation import corpus_bleu, rouge_l_f1  # noqa: E402
from projects.image_captioning.inference import generate_caption_from_backbone_features  # noqa: E402
from projects.image_captioning.model import CaptioningModel  # noqa: E402
from projects.image_captioning.training import (  # noqa: E402
    EarlyStopping,
    TrainingConfig,
    make_plateau_scheduler,
    run_caption_epoch,
    save_caption_checkpoint,
)
from projects.image_captioning.transforms import describe_image_preprocessing  # noqa: E402
from projects.image_captioning.vocab import Vocabulary, tokenize  # noqa: E402


VARIANTS = [
    {'name': 'small_lstm_256', 'embed_size': 256, 'hidden_size': 256, 'num_layers': 1, 'dropout': 0.0},
    {'name': 'baseline_lstm_512', 'embed_size': 256, 'hidden_size': 512, 'num_layers': 1, 'dropout': 0.0},
    {'name': 'two_layer_lstm_dropout', 'embed_size': 256, 'hidden_size': 512, 'num_layers': 2, 'dropout': 0.3},
]


def select_device(preferred: str) -> torch.device:
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_vocabulary(train_captions: pd.DataFrame, min_freq: int) -> tuple[Vocabulary, int]:
    vocabulary = Vocabulary(min_freq=min_freq)
    vocabulary.fit(train_captions['caption'].tolist())
    caption_lengths = train_captions['caption'].map(lambda text: len(tokenize(text)))
    max_caption_length = int(caption_lengths.quantile(0.95).round()) + 2
    return vocabulary, max_caption_length


def evaluate_sample_generation(model,
                               vocabulary: Vocabulary,
                               test_captions: pd.DataFrame,
                               features_dir: Path,
                               device: torch.device,
                               max_caption_length: int,
                               max_images: int) -> dict[str, float]:
    references = []
    hypotheses = []
    for image_name, captions in list(test_captions.groupby('image')['caption'].apply(list).items())[:max_images]:
        feature_path = features_dir / f'{Path(image_name).stem}.pt'
        features = torch.load(feature_path, map_location='cpu', weights_only=True)
        generated = generate_caption_from_backbone_features(
            model,
            features,
            vocabulary,
            max_length=max_caption_length,
            device=device,
            decoding='greedy',
        )
        references.append([tokenize(caption) for caption in captions])
        hypotheses.append(tokenize(generated))
    return {
        'sample_bleu_1': corpus_bleu(references, hypotheses, max_order=1),
        'sample_bleu_4': corpus_bleu(references, hypotheses, max_order=4),
        'sample_rouge_l': rouge_l_f1(references, hypotheses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run cached-feature architecture experiments.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'DATA' / 'flickr8k')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports' / 'model' / 'architecture_experiments.csv')
    parser.add_argument('--checkpoint-dir', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'experiments')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gradient-clip', type=float, default=5.0)
    parser.add_argument('--min-word-frequency', type=int, default=5)
    parser.add_argument('--sample-metric-images', type=int, default=100)
    parser.add_argument('--variant', action='append', choices=[variant['name'] for variant in VARIANTS])
    args = parser.parse_args()

    device = select_device(args.device)
    splits_dir = args.data_dir / 'splits'
    features_dir = args.data_dir / 'features' / 'vgg16_pool7'
    train_captions = pd.read_csv(splits_dir / 'train.csv')
    val_captions = pd.read_csv(splits_dir / 'validation.csv')
    test_captions = pd.read_csv(splits_dir / 'test.csv')
    vocabulary, max_caption_length = build_vocabulary(train_captions, args.min_word_frequency)
    collate_fn = make_caption_collate_fn(vocabulary.pad_idx)

    train_dataset = Flickr8kFeatureCaptionDataset(features_dir, train_captions, vocabulary, max_caption_length)
    val_dataset = Flickr8kFeatureCaptionDataset(features_dir, val_captions, vocabulary, max_caption_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    selected = [variant for variant in VARIANTS if args.variant is None or variant['name'] in args.variant]
    results = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for variant in selected:
        print(f'Running variant: {variant["name"]}', flush=True)
        torch.manual_seed(42)
        model = CaptioningModel(
            vocab_size=len(vocabulary),
            embed_size=variant['embed_size'],
            hidden_size=variant['hidden_size'],
            num_layers=variant['num_layers'],
            dropout=variant['dropout'],
            freeze_encoder=True,
            pretrained_encoder=False,
        ).to(device)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
        training_config = TrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate, gradient_clip=args.gradient_clip)
        scheduler = make_plateau_scheduler(optimizer, training_config)
        stopper = EarlyStopping(patience=2)
        history = []
        best_validation_loss = float('inf')
        checkpoint_path = args.checkpoint_dir / f'{variant["name"]}.pt'

        for epoch in range(1, args.epochs + 1):
            train_loss = run_caption_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                gradient_clip=args.gradient_clip,
                log_every=250,
            )
            validation_loss = run_caption_epoch(model, val_loader, criterion, device)
            scheduler.step(validation_loss)
            history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss})
            print(f'  epoch {epoch}: train={train_loss:.4f}, val={validation_loss:.4f}', flush=True)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_caption_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    vocabulary,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    model_config={
                        'embed_size': variant['embed_size'],
                        'hidden_size': variant['hidden_size'],
                        'num_layers': variant['num_layers'],
                        'dropout': variant['dropout'],
                        'freeze_encoder': True,
                    },
                    max_caption_length=max_caption_length,
                    history=history,
                    training_config=training_config,
                    preprocessing_config=describe_image_preprocessing(),
                )

            if stopper.step(validation_loss):
                break

        sample_metrics = evaluate_sample_generation(
            model,
            vocabulary,
            test_captions,
            features_dir,
            device,
            max_caption_length,
            args.sample_metric_images,
        )
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        results.append({
            **variant,
            'best_validation_loss': best_validation_loss,
            'epochs_run': len(history),
            'trainable_parameters': trainable_parameters,
            **sample_metrics,
        })

    result_frame = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output, index=False)
    print(result_frame.to_string(index=False), flush=True)
    print(f'Saved architecture experiments to {args.output}', flush=True)


if __name__ == '__main__':
    main()

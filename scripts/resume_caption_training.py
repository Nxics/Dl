"""Resume VGG16+LSTM captioning training from the current best checkpoint."""

from __future__ import annotations

import argparse
import shutil
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
from projects.image_captioning.evaluation import load_caption_checkpoint  # noqa: E402
from projects.image_captioning.training import (  # noqa: E402
    EarlyStopping,
    TrainingConfig,
    make_plateau_scheduler,
    run_caption_epoch,
    save_caption_checkpoint,
)
from projects.image_captioning.transforms import describe_image_preprocessing  # noqa: E402


def select_device(preferred: str) -> torch.device:
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer,
                                    device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def resume_training(args: argparse.Namespace) -> Path:
    data_dir = args.data_dir
    splits_dir = data_dir / 'splits'
    features_dir = data_dir / 'features' / 'vgg16_pool7'
    checkpoint_path = args.checkpoint
    latest_path = checkpoint_path.with_name('latest_model.pt')
    backup_path = checkpoint_path.with_name('best_model_before_resume.pt')

    if checkpoint_path.exists() and not backup_path.exists():
        shutil.copy2(checkpoint_path, backup_path)
        print(f'Backup saved: {backup_path}', flush=True)

    device = select_device(args.device)
    print(f'Device: {device}', flush=True)

    model, vocabulary, checkpoint = load_caption_checkpoint(checkpoint_path, device)
    start_epoch = int(checkpoint.get('epoch', 0))
    best_validation_loss = float(checkpoint.get('validation_loss', 'inf'))
    history = list(checkpoint.get('history', []))
    model_config = checkpoint.get('model_config', {})
    max_caption_length = int(checkpoint.get('max_caption_length', args.max_caption_length))

    train_captions = pd.read_csv(splits_dir / 'train.csv')
    val_captions = pd.read_csv(splits_dir / 'validation.csv')

    collate_fn = make_caption_collate_fn(vocabulary.pad_idx)
    train_dataset = Flickr8kFeatureCaptionDataset(
        features_dir,
        train_captions,
        vocabulary,
        max_caption_length,
    )
    val_dataset = Flickr8kFeatureCaptionDataset(
        features_dir,
        val_captions,
        vocabulary,
        max_caption_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume_optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate
                group['weight_decay'] = args.weight_decay
            _move_optimizer_state_to_device(optimizer, device)
            print('Optimizer state resumed.', flush=True)
        except ValueError as error:
            print(f'Optimizer state was not resumed: {error}', flush=True)

    training_config = TrainingConfig(
        epochs=args.target_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        min_delta=args.min_delta,
    )
    scheduler = make_plateau_scheduler(optimizer, training_config)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        best_loss=best_validation_loss,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
    preprocessing_config = describe_image_preprocessing(args.image_size)

    print(f'Start epoch: {start_epoch}', flush=True)
    print(f'Current best validation loss: {best_validation_loss:.4f}', flush=True)
    print(f'Target epoch: {args.target_epoch}', flush=True)
    print(f'Train rows: {len(train_dataset):,} | validation rows: {len(val_dataset):,}', flush=True)

    for epoch in range(start_epoch + 1, args.target_epoch + 1):
        train_loss = run_caption_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
            log_every=args.log_every,
        )
        validation_loss = run_caption_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step(validation_loss)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'learning_rate': current_lr,
        }
        history.append(epoch_metrics)

        print(
            f'Epoch {epoch:02d}/{args.target_epoch:02d} | '
            f'train loss: {train_loss:.4f} | '
            f'validation loss: {validation_loss:.4f} | '
            f'lr: {current_lr:.2e}',
            flush=True,
        )

        save_caption_checkpoint(
            latest_path,
            model,
            optimizer,
            vocabulary,
            epoch=epoch,
            validation_loss=validation_loss,
            model_config=model_config,
            max_caption_length=max_caption_length,
            history=history,
            training_config=training_config,
            preprocessing_config=preprocessing_config,
            notes='Latest checkpoint from resumed training.',
        )

        if validation_loss < best_validation_loss - args.min_delta:
            best_validation_loss = validation_loss
            save_caption_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                vocabulary,
                epoch=epoch,
                validation_loss=validation_loss,
                model_config=model_config,
                max_caption_length=max_caption_length,
                history=history,
                training_config=training_config,
                preprocessing_config=preprocessing_config,
                notes='Best checkpoint from resumed training.',
            )
            print(f'  ✓ New best checkpoint saved: {checkpoint_path}', flush=True)

        if early_stopping.step(validation_loss):
            print('Early stopping triggered.', flush=True)
            break

    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Resume Flickr8k captioning training.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'DATA' / 'flickr8k')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'best_model.pt')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--target-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--gradient-clip', type=float, default=5.0)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument('--scheduler-patience', type=int, default=2)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--max-caption-length', type=int, default=20)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--resume-optimizer', action='store_true')
    args = parser.parse_args()

    resume_training(args)


if __name__ == '__main__':
    main()

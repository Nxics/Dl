from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from projects.image_captioning.vocab import Vocabulary


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a full captioning training run."""

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_delta: float = 1e-4


@dataclass
class EarlyStopping:
    """Track validation loss and decide when training should stop."""

    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float | None = None
    bad_epochs: int = 0

    def step(self, validation_loss: float) -> bool:
        """Return True when training should stop."""

        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def make_plateau_scheduler(optimizer: torch.optim.Optimizer,
                           config: TrainingConfig
                           ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create a validation-loss scheduler for captioning training."""

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )


def run_caption_epoch(model: nn.Module,
                      data_loader: Iterable,
                      criterion: nn.Module,
                      device: torch.device | str,
                      optimizer: torch.optim.Optimizer | None = None,
                      gradient_clip: float | None = None,
                      max_batches: int | None = None,
                      log_every: int | None = None) -> float:
    """Run one training or validation epoch and return token-weighted loss."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_tokens = 0

    for batch_index, (images, captions) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        captions = captions.to(device)
        decoder_inputs = captions[:, :-1]
        targets = captions[:, 1:]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images, decoder_inputs)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            if is_training:
                loss.backward()
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        ignore_index = getattr(criterion, 'ignore_index', -100)
        token_count = int(targets.ne(ignore_index).sum().item())
        total_loss += float(loss.detach().cpu()) * token_count
        total_tokens += token_count

        if log_every is not None and (batch_index + 1) % log_every == 0:
            running_loss = total_loss / total_tokens
            print(
                f'  batch {batch_index + 1:,} | '
                f'average loss: {running_loss:.4f}',
                flush=True,
            )

    if total_tokens == 0:
        raise ValueError('The data loader produced no target tokens')

    return total_loss / total_tokens


def save_caption_checkpoint(path: str | Path,
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            vocabulary: Vocabulary,
                            epoch: int,
                            validation_loss: float,
                            model_config: dict[str, int | bool],
                            max_caption_length: int,
                            history: list[dict[str, float | int]],
                            training_config: TrainingConfig | None = None,
                            preprocessing_config: dict[str, object] | None = None,
                            notes: str | None = None) -> Path:
    """Save a checkpoint compatible with the Streamlit application."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'validation_loss': validation_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'training_config': (
            asdict(training_config) if training_config is not None else None
        ),
        'preprocessing_config': preprocessing_config,
        'max_caption_length': max_caption_length,
        'history': history,
        'notes': notes,
        'vocabulary': {
            'min_freq': vocabulary.min_freq,
            'token_to_idx': vocabulary.token_to_idx,
            'idx_to_token': vocabulary.idx_to_token,
        },
    }
    torch.save(checkpoint, path)
    return path

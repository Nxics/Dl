import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from projects.image_captioning.training import (
    EarlyStopping,
    TrainingConfig,
    make_plateau_scheduler,
    run_caption_epoch,
    save_caption_checkpoint,
)
from projects.image_captioning.vocab import Vocabulary


class TinyCaptionModel(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.output = nn.Linear(8, vocab_size)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        del images
        return self.output(self.embedding(captions))


class TestTraining(unittest.TestCase):

    def test_when_training_epoch_runs_then_returns_finite_loss(self):
        # Arrange
        images = torch.randn(4, 3, 4, 4)
        captions = torch.tensor([
            [1, 4, 5, 2, 0],
            [1, 5, 2, 0, 0],
            [1, 4, 4, 2, 0],
            [1, 5, 5, 2, 0],
        ])
        loader = DataLoader(TensorDataset(images, captions), batch_size=2)
        model = TinyCaptionModel(vocab_size=6)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Act
        loss = run_caption_epoch(
            model,
            loader,
            criterion,
            device='cpu',
            optimizer=optimizer,
            gradient_clip=1.0,
        )

        # Assert
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_when_checkpoint_saved_then_required_state_is_present(self):
        # Arrange
        model = TinyCaptionModel(vocab_size=6)
        optimizer = torch.optim.Adam(model.parameters())
        vocabulary = Vocabulary()
        vocabulary.fit(['a dog'])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'model.pt'

            # Act
            save_caption_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                vocabulary,
                epoch=1,
                validation_loss=2.5,
                model_config={'embed_size': 8},
                max_caption_length=10,
                history=[{'epoch': 1, 'train_loss': 3.0, 'validation_loss': 2.5}],
            )
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Assert
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('vocabulary', checkpoint)
            self.assertEqual(checkpoint['epoch'], 1)

    def test_when_early_stopping_patience_is_reached_then_stops(self):
        # Arrange
        stopper = EarlyStopping(patience=2, min_delta=0.01)

        # Act / Assert
        self.assertFalse(stopper.step(2.0))
        self.assertFalse(stopper.step(1.995))
        self.assertTrue(stopper.step(1.994))

    def test_when_scheduler_created_then_uses_training_config(self):
        # Arrange
        model = TinyCaptionModel(vocab_size=6)
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig(scheduler_factor=0.25, scheduler_patience=3)

        # Act
        scheduler = make_plateau_scheduler(optimizer, config)

        # Assert
        self.assertEqual(scheduler.factor, 0.25)
        self.assertEqual(scheduler.patience, 3)

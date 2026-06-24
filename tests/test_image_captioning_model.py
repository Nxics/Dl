import unittest

import torch

from projects.image_captioning.model import LstmDecoder


class TestLstmDecoder(unittest.TestCase):

    def test_when_decoder_runs_then_returns_one_prediction_per_input_token(self):
        # Arrange
        decoder = LstmDecoder(vocab_size=20, embed_size=8, hidden_size=16)
        image_features = torch.randn(2, 8)
        captions = torch.randint(0, 20, (2, 5))

        # Act
        logits = decoder(image_features, captions)

        # Assert
        self.assertEqual(tuple(logits.shape), (2, 5, 20))

    def test_when_training_targets_created_then_logits_and_targets_align(self):
        # Arrange
        decoder = LstmDecoder(vocab_size=20, embed_size=8, hidden_size=16)
        image_features = torch.randn(2, 8)
        captions = torch.randint(0, 20, (2, 5))

        # Act
        logits = decoder(image_features, captions[:, :-1])
        targets = captions[:, 1:]

        # Assert
        self.assertEqual(tuple(logits.shape[:2]), tuple(targets.shape))

    def test_when_cached_backbone_features_used_then_model_skips_image_encoder(self):
        # Arrange
        from projects.image_captioning.model import CaptioningModel

        model = CaptioningModel(
            vocab_size=20,
            embed_size=8,
            hidden_size=16,
            pretrained_encoder=False,
        )
        cached_features = torch.randn(2, 512 * 7 * 7)
        captions = torch.randint(0, 20, (2, 5))

        # Act
        logits = model(cached_features, captions)

        # Assert
        self.assertEqual(tuple(logits.shape), (2, 5, 20))

    def test_when_two_layer_decoder_created_then_dropout_is_enabled(self):
        # Arrange / Act
        decoder = LstmDecoder(
            vocab_size=20,
            embed_size=8,
            hidden_size=16,
            num_layers=2,
            dropout=0.3,
        )

        # Assert
        self.assertEqual(decoder.lstm.dropout, 0.3)

"""Experiment definitions for the Flickr8k captioning project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Small, report-friendly description of one modeling experiment."""

    name: str
    image_preprocessing: str
    augmentation: str
    word_embedding: str
    encoder: str
    decoder: str
    expected_tradeoff: str


def get_experiment_configs() -> list[ExperimentConfig]:
    """Return the planned baseline and comparison experiments."""

    return [
        ExperimentConfig(
            name='baseline_vgg16_lstm',
            image_preprocessing='resize_224_imagenet_normalization',
            augmentation='none_for_eval_only',
            word_embedding='trainable_embedding',
            encoder='frozen_vgg16',
            decoder='one_layer_lstm',
            expected_tradeoff='stable baseline with fast training',
        ),
        ExperimentConfig(
            name='augmented_vgg16_lstm',
            image_preprocessing='resize_224_imagenet_normalization',
            augmentation='horizontal_flip_and_small_rotation',
            word_embedding='trainable_embedding',
            encoder='frozen_vgg16',
            decoder='one_layer_lstm',
            expected_tradeoff='better robustness, slightly noisier training',
        ),
        ExperimentConfig(
            name='larger_lstm_decoder',
            image_preprocessing='resize_224_imagenet_normalization',
            augmentation='horizontal_flip_and_small_rotation',
            word_embedding='trainable_embedding',
            encoder='frozen_vgg16',
            decoder='larger_hidden_size_lstm',
            expected_tradeoff='more capacity, higher overfitting risk',
        ),
        ExperimentConfig(
            name='dropout_two_layer_lstm',
            image_preprocessing='resize_224_imagenet_normalization',
            augmentation='horizontal_flip_and_small_rotation',
            word_embedding='trainable_embedding',
            encoder='frozen_vgg16',
            decoder='two_layer_lstm_with_dropout',
            expected_tradeoff='stronger language model, slower training',
        ),
        ExperimentConfig(
            name='pretrained_word_vectors',
            image_preprocessing='resize_224_imagenet_normalization',
            augmentation='horizontal_flip_and_small_rotation',
            word_embedding='word2vec_glove_or_fasttext_initialization',
            encoder='frozen_vgg16',
            decoder='one_layer_lstm',
            expected_tradeoff='better starting word space, extra preprocessing',
        ),
    ]


def summarize_experiment_configs() -> list[dict[str, str]]:
    """Return experiment configs as dictionaries for notebooks or reports."""

    return [
        {
            'name': config.name,
            'image_preprocessing': config.image_preprocessing,
            'augmentation': config.augmentation,
            'word_embedding': config.word_embedding,
            'encoder': config.encoder,
            'decoder': config.decoder,
            'expected_tradeoff': config.expected_tradeoff,
        }
        for config in get_experiment_configs()
    ]

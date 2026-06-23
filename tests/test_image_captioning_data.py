import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from projects.image_captioning.data import (
    Flickr8kCaptionDataset,
    Flickr8kFeatureCaptionDataset,
    read_flickr8k_captions,
)
from projects.image_captioning.vocab import Vocabulary


class TestReadFlickr8kCaptions(unittest.TestCase):

    def test_when_csv_has_image_and_caption_columns_then_returns_required_columns(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            captions_path = Path(temp_dir) / 'captions.csv'
            captions_path.write_text('image,caption\nimg.jpg,a dog runs\n', encoding='utf-8')

            # Act
            actual = read_flickr8k_captions(captions_path)

            # Assert
            self.assertEqual(list(actual.columns), ['image', 'caption'])
            self.assertEqual(len(actual), 1)

    def test_when_legacy_file_used_then_removes_caption_index_from_image_name(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            captions_path = Path(temp_dir) / 'captions.txt'
            captions_path.write_text('img.jpg#0\ta dog runs\n', encoding='utf-8')

            # Act
            actual = read_flickr8k_captions(captions_path)

            # Assert
            self.assertEqual(actual.iloc[0]['image'], 'img.jpg')
            self.assertEqual(actual.iloc[0]['caption'], 'a dog runs')

    def test_when_kaggle_pipe_format_used_then_returns_all_captions(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            captions_path = Path(temp_dir) / 'captions.txt'
            captions_path.write_text(
                'image_name|caption_number|caption_text\n'
                'img.jpg|0|a dog runs\n'
                'img.jpg|1|a dog jumps\n',
                encoding='utf-8',
            )

            # Act
            actual = read_flickr8k_captions(captions_path)

            # Assert
            self.assertEqual(list(actual.columns), ['image', 'caption'])
            self.assertEqual(len(actual), 2)
            self.assertEqual(actual.iloc[1]['caption'], 'a dog jumps')


class TestFlickr8kCaptionDataset(unittest.TestCase):

    def test_when_item_requested_then_returns_image_tensor_and_caption_tensor(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / 'images'
            images_dir.mkdir()
            Image.new('RGB', (8, 8), color='white').save(images_dir / 'img.jpg')
            captions_path = temp_path / 'captions.csv'
            captions_path.write_text('image,caption\nimg.jpg,a dog runs\n', encoding='utf-8')
            captions = read_flickr8k_captions(captions_path)
            vocabulary = Vocabulary()
            vocabulary.fit(captions['caption'].tolist())
            dataset = Flickr8kCaptionDataset(images_dir, captions, vocabulary, max_length=8)

            # Act
            image, caption = dataset[0]

            # Assert
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(caption.shape[0], 8)

    def test_when_caption_is_truncated_then_last_token_is_end_token(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / 'images'
            images_dir.mkdir()
            Image.new('RGB', (8, 8), color='white').save(images_dir / 'img.jpg')
            captions_path = temp_path / 'captions.csv'
            captions_path.write_text(
                'image,caption\nimg.jpg,one two three four five\n',
                encoding='utf-8',
            )
            captions = read_flickr8k_captions(captions_path)
            vocabulary = Vocabulary()
            vocabulary.fit(captions['caption'].tolist())
            dataset = Flickr8kCaptionDataset(
                images_dir,
                captions,
                vocabulary,
                max_length=4,
            )

            # Act
            _, caption = dataset[0]

            # Assert
            self.assertEqual(int(caption[-1]), vocabulary.end_idx)

    def test_when_cached_feature_requested_then_returns_feature_and_caption(self):
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            features_dir = Path(temp_dir)
            torch_feature = torch.randn(512 * 7 * 7).half()
            torch.save(torch_feature, features_dir / 'img.pt')
            captions_path = features_dir / 'captions.csv'
            captions_path.write_text(
                'image,caption\nimg.jpg,a dog runs\n',
                encoding='utf-8',
            )
            captions = read_flickr8k_captions(captions_path)
            vocabulary = Vocabulary()
            vocabulary.fit(captions['caption'].tolist())
            dataset = Flickr8kFeatureCaptionDataset(
                features_dir,
                captions,
                vocabulary,
                max_length=8,
            )

            # Act
            feature, caption = dataset[0]

            # Assert
            self.assertEqual(tuple(feature.shape), (512 * 7 * 7,))
            self.assertEqual(feature.dtype, torch.float32)
            self.assertEqual(caption.shape[0], 8)

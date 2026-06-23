import unittest

from projects.image_captioning.vocab import Vocabulary, tokenize


class TestTokenize(unittest.TestCase):

    def test_when_text_has_uppercase_and_punctuation_then_returns_lowercase_words(self):
        # Arrange
        text = 'A Dog, runs!'
        expected = ['a', 'dog', 'runs']

        # Act
        actual = tokenize(text)

        # Assert
        self.assertEqual(actual, expected)


class TestVocabulary(unittest.TestCase):

    def test_when_fit_called_then_adds_frequent_words(self):
        # Arrange
        vocabulary = Vocabulary(min_freq=2)
        captions = ['A dog runs', 'A dog jumps', 'A cat sleeps']

        # Act
        vocabulary.fit(captions)

        # Assert
        self.assertIn('a', vocabulary.token_to_idx)
        self.assertIn('dog', vocabulary.token_to_idx)
        self.assertNotIn('cat', vocabulary.token_to_idx)

    def test_when_caption_encoded_then_start_and_end_tokens_are_added(self):
        # Arrange
        vocabulary = Vocabulary()
        vocabulary.fit(['a dog runs'])

        # Act
        actual = vocabulary.encode('a dog')

        # Assert
        self.assertEqual(actual[0], vocabulary.start_idx)
        self.assertEqual(actual[-1], vocabulary.end_idx)

    def test_when_unknown_word_encoded_then_unk_token_is_used(self):
        # Arrange
        vocabulary = Vocabulary()
        vocabulary.fit(['a dog runs'])

        # Act
        actual = vocabulary.encode('spaceship', add_special_tokens=False)

        # Assert
        self.assertEqual(actual, [vocabulary.unk_idx])

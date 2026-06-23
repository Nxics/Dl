import unittest

from projects.image_captioning.evaluation import corpus_bleu, rouge_l_f1


class TestCaptionMetrics(unittest.TestCase):

    def test_when_hypothesis_matches_reference_then_scores_are_one(self):
        # Arrange
        references = [[['a', 'dog', 'runs', 'fast']]]
        hypotheses = [['a', 'dog', 'runs', 'fast']]

        # Act
        bleu = corpus_bleu(references, hypotheses, max_order=4)
        rouge = rouge_l_f1(references, hypotheses)

        # Assert
        self.assertAlmostEqual(bleu, 1.0)
        self.assertAlmostEqual(rouge, 1.0)

    def test_when_hypothesis_is_empty_then_scores_are_zero(self):
        # Arrange
        references = [[['a', 'dog', 'runs']]]
        hypotheses = [[]]

        # Act and assert
        self.assertEqual(corpus_bleu(references, hypotheses), 0.0)
        self.assertEqual(rouge_l_f1(references, hypotheses), 0.0)


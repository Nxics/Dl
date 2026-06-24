import math
from collections import Counter
from pathlib import Path

import torch

from projects.image_captioning.model import CaptioningModel
from projects.image_captioning.vocab import Vocabulary


def load_caption_checkpoint(path: str | Path,
                            device: torch.device | str = 'cpu'
                            ) -> tuple[CaptioningModel, Vocabulary, dict]:
    """Load a trained captioning model, vocabulary, and checkpoint metadata."""
    checkpoint = torch.load(Path(path), map_location=device)
    vocab_state = checkpoint['vocabulary']
    vocabulary = Vocabulary(min_freq=vocab_state.get('min_freq', 1))
    vocabulary.token_to_idx = vocab_state['token_to_idx']
    vocabulary.idx_to_token = {
        int(index): token for index, token in vocab_state['idx_to_token'].items()
    }

    config = checkpoint.get('model_config', {})
    model = CaptioningModel(
        vocab_size=len(vocabulary),
        embed_size=config.get('embed_size', 256),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 1),
        freeze_encoder=config.get('freeze_encoder', True),
        pretrained_encoder=False,
        dropout=config.get('dropout', 0.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, vocabulary, checkpoint


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index:index + order])
        for index in range(len(tokens) - order + 1)
    )


def corpus_bleu(references: list[list[list[str]]],
                hypotheses: list[list[str]],
                max_order: int = 4,
                smooth: bool = True) -> float:
    """Compute corpus BLEU with clipped n-gram precision."""
    if len(references) != len(hypotheses):
        raise ValueError('references and hypotheses must have equal length')
    if not hypotheses:
        return 0.0

    matches = [0] * max_order
    totals = [0] * max_order
    hypothesis_length = 0
    reference_length = 0

    for image_references, hypothesis in zip(references, hypotheses):
        hypothesis_length += len(hypothesis)
        reference_lengths = [len(reference) for reference in image_references]
        reference_length += min(
            reference_lengths,
            key=lambda length: (abs(length - len(hypothesis)), length),
        )

        for order in range(1, max_order + 1):
            hypothesis_ngrams = _ngrams(hypothesis, order)
            maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
            for reference in image_references:
                reference_ngrams = _ngrams(reference, order)
                for ngram, count in reference_ngrams.items():
                    maximum_reference_counts[ngram] = max(
                        maximum_reference_counts[ngram],
                        count,
                    )

            matches[order - 1] += sum(
                min(count, maximum_reference_counts[ngram])
                for ngram, count in hypothesis_ngrams.items()
            )
            totals[order - 1] += sum(hypothesis_ngrams.values())

    if hypothesis_length == 0:
        return 0.0

    precisions = []
    for match_count, total_count in zip(matches, totals):
        if smooth:
            precisions.append((match_count + 1) / (total_count + 1))
        else:
            precisions.append(match_count / total_count if total_count else 0.0)

    if any(precision == 0 for precision in precisions):
        return 0.0

    brevity_penalty = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1 - reference_length / hypothesis_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


def _lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(references: list[list[list[str]]],
               hypotheses: list[list[str]]) -> float:
    """Average best-reference ROUGE-L F1 score."""
    if len(references) != len(hypotheses):
        raise ValueError('references and hypotheses must have equal length')
    if not hypotheses:
        return 0.0

    scores = []
    for image_references, hypothesis in zip(references, hypotheses):
        best_score = 0.0
        for reference in image_references:
            if not reference or not hypothesis:
                score = 0.0
            else:
                lcs = _lcs_length(reference, hypothesis)
                precision = lcs / len(hypothesis)
                recall = lcs / len(reference)
                score = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            best_score = max(best_score, score)
        scores.append(best_score)
    return sum(scores) / len(scores)

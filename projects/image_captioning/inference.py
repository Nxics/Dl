import torch
from torch.nn import functional as F
from PIL import Image

from projects.image_captioning.vocab import Vocabulary


def _greedy_decode(model,
                   image_features: torch.Tensor,
                   vocabulary: Vocabulary,
                   max_length: int,
                   device: torch.device | str) -> str:
    generated = [vocabulary.start_idx]
    with torch.no_grad():
        for _ in range(max_length):
            captions = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model.decoder(image_features, captions)
            next_idx = int(logits[:, -1, :].argmax(dim=-1).item())
            if next_idx == vocabulary.end_idx:
                break
            generated.append(next_idx)
    return vocabulary.decode(generated)


def _beam_search_decode(model,
                        image_features: torch.Tensor,
                        vocabulary: Vocabulary,
                        max_length: int,
                        device: torch.device | str,
                        beam_size: int = 3,
                        length_penalty: float = 0.7) -> str:
    """Generate a caption by keeping the best partial sequences."""
    if beam_size <= 1:
        return _greedy_decode(model, image_features, vocabulary, max_length, device)

    beams: list[tuple[list[int], float, bool]] = [
        ([vocabulary.start_idx], 0.0, False)
    ]
    blocked_indices = [
        vocabulary.pad_idx,
        vocabulary.start_idx,
        vocabulary.unk_idx,
    ]

    with torch.no_grad():
        for _ in range(max_length):
            candidates: list[tuple[list[int], float, bool]] = []

            for sequence, score, ended in beams:
                if ended:
                    candidates.append((sequence, score, True))
                    continue

                captions = torch.tensor([sequence], dtype=torch.long, device=device)
                logits = model.decoder(image_features, captions)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                log_probs[blocked_indices] = float('-inf')

                top_scores, top_indices = torch.topk(log_probs, k=beam_size)
                for next_score, next_idx in zip(top_scores.tolist(), top_indices.tolist()):
                    next_sequence = sequence + [int(next_idx)]
                    next_ended = int(next_idx) == vocabulary.end_idx
                    candidates.append((next_sequence, score + float(next_score), next_ended))

            def normalized_score(candidate: tuple[list[int], float, bool]) -> float:
                sequence, score, _ = candidate
                generated_length = max(1, len(sequence) - 1)
                return score / (generated_length ** length_penalty)

            beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_size]
            if all(ended for _, _, ended in beams):
                break

    best_sequence = max(
        beams,
        key=lambda candidate: candidate[1] / (max(1, len(candidate[0]) - 1) ** length_penalty),
    )[0]
    return vocabulary.decode(best_sequence)


def generate_caption(model,
                     image: Image.Image,
                     vocabulary: Vocabulary,
                     transform,
                     max_length: int = 30,
                     device: str = 'cpu',
                     decoding: str = 'greedy',
                     beam_size: int = 3) -> str:
    """Generate a caption with greedy or beam-search decoding."""
    model.eval()
    model.to(device)

    image_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encoder(image_tensor)
    if decoding == 'beam':
        return _beam_search_decode(
            model,
            image_features,
            vocabulary,
            max_length,
            device,
            beam_size=beam_size,
        )
    return _greedy_decode(
        model,
        image_features,
        vocabulary,
        max_length,
        device,
    )


def generate_caption_from_backbone_features(
        model,
        backbone_features: torch.Tensor,
        vocabulary: Vocabulary,
        max_length: int = 30,
        device: torch.device | str = 'cpu',
        decoding: str = 'greedy',
        beam_size: int = 3) -> str:
    """Generate a caption from cached pooled VGG features."""
    model.eval()
    model.to(device)
    backbone_features = backbone_features.to(device).float()
    if backbone_features.ndim == 1:
        backbone_features = backbone_features.unsqueeze(0)
    with torch.no_grad():
        image_features = model.encoder.project_backbone_features(backbone_features)
    if decoding == 'beam':
        return _beam_search_decode(
            model,
            image_features,
            vocabulary,
            max_length,
            device,
            beam_size=beam_size,
        )
    return _greedy_decode(
        model,
        image_features,
        vocabulary,
        max_length,
        device,
    )

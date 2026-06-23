import torch
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


def generate_caption(model,
                     image: Image.Image,
                     vocabulary: Vocabulary,
                     transform,
                     max_length: int = 30,
                     device: str = 'cpu') -> str:
    """Generate a caption with greedy decoding."""
    model.eval()
    model.to(device)

    image_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encoder(image_tensor)
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
        device: torch.device | str = 'cpu') -> str:
    """Generate a caption from cached pooled VGG features."""
    model.eval()
    model.to(device)
    backbone_features = backbone_features.to(device).float()
    if backbone_features.ndim == 1:
        backbone_features = backbone_features.unsqueeze(0)
    with torch.no_grad():
        image_features = model.encoder.project_backbone_features(backbone_features)
    return _greedy_decode(
        model,
        image_features,
        vocabulary,
        max_length,
        device,
    )

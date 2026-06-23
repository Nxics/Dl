import torch
from torch import nn
from torchvision import models


class VggEncoder(nn.Module):
    """VGG encoder that maps an image to one embedding vector."""

    def __init__(self,
                 embed_size: int = 256,
                 freeze: bool = True,
                 pretrained: bool = True) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.projection = nn.Linear(512 * 7 * 7, embed_size)
        self.activation = nn.ReLU()

        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def extract_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled VGG features before the trainable projection."""
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, start_dim=1)

    def project_backbone_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project cached VGG features into the decoder embedding space."""
        x = self.projection(features)
        return self.activation(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.project_backbone_features(
            self.extract_backbone_features(images)
        )


class LstmDecoder(nn.Module):
    """LSTM decoder predicting next-word logits."""

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 256,
                 hidden_size: int = 512,
                 num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        caption_embeddings = self.embedding(captions)
        image_features = image_features.unsqueeze(1)
        lstm_inputs = torch.cat((image_features, caption_embeddings), dim=1)
        outputs, _ = self.lstm(lstm_inputs)
        # Drop the output produced directly from the image. Each remaining
        # position predicts the token following the corresponding caption token.
        return self.output(outputs[:, 1:])


class CaptioningModel(nn.Module):
    """VGG + LSTM image captioning model."""

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 256,
                 hidden_size: int = 512,
                 num_layers: int = 1,
                 freeze_encoder: bool = True,
                 pretrained_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = VggEncoder(embed_size=embed_size,
                                  freeze=freeze_encoder,
                                  pretrained=pretrained_encoder)
        self.decoder = LstmDecoder(vocab_size=vocab_size,
                                   embed_size=embed_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers)

    def forward(self,
                images_or_backbone_features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        if images_or_backbone_features.ndim == 2:
            image_features = self.encoder.project_backbone_features(
                images_or_backbone_features
            )
        elif images_or_backbone_features.ndim == 4:
            image_features = self.encoder(images_or_backbone_features)
        else:
            raise ValueError(
                'Expected image tensors [B, C, H, W] or cached features [B, F]'
            )
        return self.decoder(image_features, captions)

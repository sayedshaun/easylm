import torch
from torch import nn
from typing import Union
import torch.nn.functional as F
from langtrain.config.config import VITConfig
from langtrain.nn.nn import PatchEmbedding, TransformerEncoderBlock



class VITImageClassifier(torch.nn.Module):
    def __init__(self, config: VITConfig) -> None:
        super(VITImageClassifier, self).__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.color_channels,
            embed_dim=config.hidden_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.hidden_size, 
                n_heads=config.num_heads,
                norm_epsilon=config.norm_epsilon,
                dropout=config.dropout) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, inputs: torch.Tensor, labels: Union[torch.Tensor, None] = None) -> torch.Tensor:
        B = inputs.shape[0]
        inputs = self.patch_embed(inputs)  # (B, num_patches, embed_dim)
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        inputs = torch.cat((cls_token, inputs), dim=1)  # (B, num_patches+1, embed_dim)
        # Add positional embedding
        inputs = inputs + self.pos_embed
        inputs = self.dropout(inputs)
        # Process through transformer blocks
        for block in self.blocks:
            inputs = block(inputs)
        # Classification
        inputs = self.norm(inputs)
        logits = self.classifier(inputs[:, 0])
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss
        else:
            return logits
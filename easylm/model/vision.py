import torch
from easylm.config import VITConfig
from easylm.nn import (
    Dropout, 
    LayerNormalization, 
    Linear, 
    PatchEmbedding, 
    TransformerEncoderBlock
)


class VisionTransformer(torch.nn.Module):
    def __init__(self, config: VITConfig) -> None:
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.color_channels,
            embed_dim=config.hidden_size
        )
        self.pos_embed = torch.nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, config.hidden_size))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.hidden_size, 
                n_heads=config.num_heads,
                norm_epsilon=config.norm_epsilon,
                dropout=config.dropout) for _ in range(config.num_layers)
        ])
        self.norm = LayerNormalization(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(0.1)
        self.head = Linear(config.hidden_size, config.num_classes)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        X = self.patch_embed(X)  # (B, num_patches, embed_dim)
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        X = torch.cat((cls_token, X), dim=1)  # (B, num_patches+1, embed_dim)
        # Add positional embedding
        X = X + self.pos_embed
        X = self.dropout(X)
        # Process through transformer blocks
        for block in self.blocks:
            X = block(X)
        # Classification
        X = self.norm(X)
        return self.head(X[:, 0])
    


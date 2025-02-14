import math
from typing import Tuple, Union
import warnings
import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_bias = bias
        self.w = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.b = nn.Parameter(torch.empty(out_features))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.w)
        if self.is_bias:
            nn.init.zeros_(self.b)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.w.to(X.device)
        self.b.to(X.device)
        if self.is_bias:
            return X @ self.w.T + self.b
        else:
            return X @ self.w.T


class Softmax(nn.Module):
    def __init__(self, dim: int) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        exp_X = torch.exp(X - torch.max(X, dim=self.dim, keepdim=True).values)
        return exp_X / torch.sum(exp_X, dim=self.dim, keepdim=True)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super(SiLU, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.sigmoid(X)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))
    

class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clone()  # Ensure we don't modify input in-place
        X[X < 0] = 0   # Set negative values to 0
        return X


class Dropout(nn.Module):
    def __init__(self, dropout: float) -> None:
        super(Dropout, self).__init__()
        self.dropout = dropout

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.rand(X.shape) > self.dropout
            return X * mask / (1 - self.dropout)
        else:
            return X


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        self.weight.to(X.device)
        if X.type != torch.long:
            X = X.long()
        return self.weight[X]
    


class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, seq_len: int, dropout: float) -> None:
        super(PositionalEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.word_encoding = Embedding(vocab_size, hidden_size)
        self.position_encoding = Embedding(seq_len, hidden_size)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, X:torch.Tensor)->torch.Tensor:
        w_embedding = self.word_encoding(X)
        positions = torch.arange(X.shape[1]).unsqueeze(0).to(X.device)
        p_embedding = self.position_encoding(positions)
        embeddings = w_embedding + p_embedding
        return self.relu(self.dropout(embeddings))


class TransformerMultiheadAttention(nn.Module):
    def __init__(self,  hidden_size: int, num_heads: int) -> None:
        super(TransformerMultiheadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = Linear(hidden_size,hidden_size, bias=True)
        self.softmax = Softmax(dim=-1)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        N, L, D = Q.shape
        Q, K, V = self.q_proj(Q), self.k_proj(K), self.v_proj(V)
        Q = Q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        score = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        weights = self.softmax(score)
        attention = weights @ V

        output = attention.transpose(1, 2).contiguous().view(N, L, D)
        return self.out_proj(output)


class LayerNormalization(nn.Module):
    def __init__(self, hidden_size:int, epsilon:float=1e-5) -> None:
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.empty(hidden_size))
        self.beta = nn.Parameter(torch.empty(hidden_size))
        
    def forward(self, X:torch.Tensor)->torch.Tensor:
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        norm_X = (X-mean) / (std + self.epsilon)
        return self.gamma * norm_X + self.beta
    

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
    
    def net(self, X: torch.Tensor) -> torch.Tensor:
        net =  nn.Sequential(
            Linear(self.hidden_size, self.intermediate_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.intermediate_size, self.hidden_size),
            Dropout(self.dropout)
        )
        return net(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, norm_epsilon: float, dropout: float) -> None:
        super(TransformerDecoderBlock, self).__init__()
        self.mha = TransformerMultiheadAttention(hidden_size, num_heads)
        self.norm_1 = LayerNormalization(hidden_size, norm_epsilon)
        self.norm_2 = LayerNormalization(hidden_size, norm_epsilon)
        self.mlp = FeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(self, X:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        attention= self.mha(X, X, X, mask)
        attention = self.norm_1(attention + X)
        output = self.mlp(attention)
        return self.norm_2(output + attention)
    

class TransformerEncoderBlock(TransformerDecoderBlock):
    def __init__(self, hidden_size: int, num_heads: int, norm_epsilon: float, dropout: float) -> None:
        super(TransformerEncoderBlock, self).__init__(hidden_size, num_heads, norm_epsilon, dropout)
        self.mha = TransformerMultiheadAttention(hidden_size, num_heads)

    def forward(self, X: torch.Tensor, paddding_mask: torch.Tensor = None) -> torch.Tensor:
        attention = self.mha(X, X, X, paddding_mask)
        attention = self.norm_1(attention + X)
        output = self.mlp(attention)
        return self.norm_2(output + attention)


class RMSNormalization(nn.Module):
    def __init__(self, dim: int, epsilon: float) -> None:
        super(RMSNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.epsilon = epsilon
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, RMSNormalization):
            nn.init.ones_(self.gamma)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(X**2, dim=-1, keepdim=True))
        norm = self.gamma * X / (rms + self.epsilon)
        return norm
    

class LlamaFeedForward(FeedForward):
    def net(self, X: torch.Tensor) -> torch.Tensor:
        net = nn.Sequential(
            Linear(self.hidden_size, self.intermediate_size),
            SiLU(),
            Linear(self.intermediate_size, self.hidden_size)
        )
        return net(X)


class LlamaAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super(LlamaAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        self.query_proj = Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, X: torch.Tensor, position_ids: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        N, S, H = X.shape # batch, seq_len, hidden_size
        # Project to Q, K, V and reshape for multi-head attention
        Q = self.query_proj(X).view(N, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(X).view(N, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(X).view(N, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (transpose to match apply_rope input)
        q_rotated, k_rotated = LlamaAttention.apply_rope(
            query=Q.transpose(1, 2),  # (batch, seq_len, num_heads, head_dim)
            key=K.transpose(1, 2),
            position_ids=position_ids,
            dim=self.head_dim,
        )
        Q = q_rotated.transpose(1, 2)  # Back to (batch, num_heads, seq_len, head_dim)
        K = k_rotated.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(N, S, H)
        return self.out_proj(context)
    
    @staticmethod
    def apply_rope(query: torch.Tensor, key: torch.Tensor, 
        position_ids: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = query.dtype
        position_ids = position_ids.to(query.device)
        # Split dimensions into two halves for rotation
        (q1, q2), (k1, k2) = query.chunk(2, dim=-1), key.chunk(2, dim=-1)
        # Compute theta values (corrected with 2 * j / dim)
        j = torch.arange(0, dim // 2, dtype=dtype, device=query.device)
        theta = 1.0 / (10000 ** (2 * j / dim))  # (dim//2,)
        # Position angles (outer product of position_ids and theta)
        angles = position_ids[:, None].float() * theta[None, :]  # (seq_len, dim//2)
        cos_theta = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
        sin_theta = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
        # Rotate q and k
        q_1_rotated = q1 * cos_theta - q2 * sin_theta
        q_2_rotated = q1 * sin_theta + q2 * cos_theta
        k_1_rotated = k1 * cos_theta - k2 * sin_theta
        k_2_rotated = k1 * sin_theta + k2 * cos_theta
        q_rotated = torch.cat((q_1_rotated, q_2_rotated), dim=-1)
        k_rotated = torch.cat((k_1_rotated, k_2_rotated), dim=-1)
        return q_rotated, k_rotated


class LlamaBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, norm_epsilon: float) -> None:
        super(LlamaBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_epsilon = norm_epsilon
        self.attention = LlamaAttention(hidden_size, num_heads)
        self.mlp = LlamaFeedForward(hidden_size, hidden_size * 4, dropout)
        self.norm1 = RMSNormalization(hidden_size, norm_epsilon)
        self.norm2 = RMSNormalization(hidden_size, norm_epsilon)
        self.norm3 = RMSNormalization(hidden_size, norm_epsilon)
        self.dropout = Dropout(dropout)

    def forward(self, X: torch.Tensor, position_ids: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        norm_1 = self.norm1(X)
        attention = self.attention(norm_1, position_ids)
        norm_2 = self.norm2(attention)
        mlp = self.mlp(norm_2)
        return mlp + attention


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.flatten = nn.Flatten(2, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        X = self.proj(X)  # (B, embed_dim, H/patch_size, W/patch_size)
        X = self.flatten(X)  # (B, embed_dim, num_patches)
        return X.transpose(1, 2)  # (B, num_patches, embed_dim)
    



__all__ = [
    "Linear",
    "Softmax",
    "SiLU",
    "ReLU",
    "Dropout",
    "Embedding",
    "PositionalEmbeddings",
    "TransformerMultiheadAttention",
    "LayerNormalization",
    "FeedForward",
    "TransformerDecoderBlock",
    "TransformerEncoderBlock",
    "RMSNormalization",
    "LlamaFeedForward",
    "LlamaAttention",
    "LlamaBlock",
    "PatchEmbedding"
]
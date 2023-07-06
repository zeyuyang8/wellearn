import torch.nn as nn
import math
import torch
import torch.nn.functional as F

# TODO: Figure out why do we need `timestep_embedding`
def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: A 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim: The dimension of the output.
        max_period: Controls the minimum frequency of the embeddings.
    
    Returns:
        An [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _make_nn_module(module_type, *args):
    if isinstance(module_type, str):
        return getattr(nn, module_type)(*args)
    else:
        return module_type(*args)

class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(self, *, d_in: int, d_out: int, bias: bool, 
                     activation, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            return self.dropout(self.activation(self.linear(x)))
    
    def __init__(self, *, d_in, d_layers, dropouts, activation, d_out):
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        
        blocks = []
        for idx, (dim, dropout) in enumerate(zip(d_layers, dropouts)):
            block = MLP.Block(d_in=d_layers[idx - 1] if idx else d_in, 
                              d_out=dim, bias=True, activation=activation, dropout=dropout)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
    
    @classmethod
    def make_baseline(cls, d_in, d_layers, dropout, d_out, activation='ReLU'):
        model = MLP(d_in=d_in, d_layers=d_layers, dropouts=dropout, 
                    activation=activation, d_out=d_out)
        return model

    def forward(self, x):
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, d_in, d_out, d_layers, dim_t=128, dropout=0.2, activation='ReLU'):
        super().__init__()
        self.dim_t = dim_t
        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, 
                                     d_out=d_in, activation=activation)
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps):
        x = x.float()
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb
        return self.mlp(x)
    
class ConditionalMLPDecoder(nn.Module):
    def __init__(self, d_in, d_out, d_layers, num_classes, dim_t=128, dropout=0.2, activation='ReLU'):
        super().__init__()
        self.num_classes = num_classes
        self.dim_t = dim_t
        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, 
                                     d_out=d_in, activation=activation)
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.label_emb = nn.Embedding(self.num_classes, dim_t)
    
    def forward(self, x, timesteps, y):
        x = x.float()
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)

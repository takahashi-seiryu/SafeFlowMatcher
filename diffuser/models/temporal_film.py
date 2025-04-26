import torch
import torch.nn as nn

class TemporalFiLM(nn.Module):
    """
    Temporal Feature-wise Linear Modulation (FiLM) layer
    Modulates features based on time embeddings for better temporal guidance
    """
    def __init__(self, feature_dim, time_embed_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * feature_dim)  # 2x for scale and shift
        )
        
    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Input features [batch_size, feature_dim]
            t (torch.Tensor): Time step [batch_size]
        Returns:
            torch.Tensor: Modulated features [batch_size, feature_dim]
        """
        # Get time embeddings directly projected to feature dimension
        time_emb = self.time_mlp(t.unsqueeze(-1))  # [batch_size, 2 * feature_dim]
        scale, shift = time_emb.chunk(2, dim=-1)    # Each [batch_size, feature_dim]
        
        # Apply modulation
        return x * (1 + scale) + shift

import torch
import torch.nn as nn
import numpy as np

class TemporalMLP(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        hidden_dim=128,
        n_hidden=3,
        dropout_rate=0.1,
        act='mish',
        time_embed_dim=128,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Activation function
        if act == 'mish':
            self.act = nn.Mish()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            raise NotImplementedError(f'Unknown activation function: {act}')

        # Input projection
        input_dim = horizon * transition_dim  # Full flattened input dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
        )

        # Hidden layers with residual connections
        self.layers = nn.ModuleList([])
        for i in range(n_hidden):
            self.layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ]))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, horizon * transition_dim)

    def forward(self, x, cond, t):
        """
        Args:
            x: [batch_size, horizon, transition_dim] input state
            cond: conditioning (not used in base model)
            t: [batch_size] time steps
        Returns:
            [batch_size, horizon, transition_dim] velocity prediction
        """
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)  # [batch_size, horizon * transition_dim]
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [batch_size, time_embed_dim]
        
        # Initial projection
        h = self.input_proj(x_flat)  # [batch_size, hidden_dim]
        
        # Apply hidden layers with residual connections
        for linear, dropout, norm in self.layers:
            # Combine time embedding with hidden state
            h_time = h + t_emb  # Add time information
            
            # Residual block
            h_res = self.act(linear(h_time))
            h_res = dropout(h_res)
            h_res = norm(h_res)
            
            # Residual connection
            h = h + h_res
        
        # Output projection and reshape
        out = self.output_proj(h)  # [batch_size, horizon * transition_dim]
        out = out.reshape(batch_size, self.horizon, self.transition_dim)
        
        return out

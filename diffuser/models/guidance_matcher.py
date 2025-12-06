import torch
import torch.nn as nn
import math


class GuidanceMatcher:
    """
    Class implementing the Cov-G method.
    """
    def __init__(
        self, 
        model: nn.Module,
        action_dim: int,
        model_z: nn.Module = None,
        model_v: nn.Module = None,
        scale: float = 1.0,
        guidance_type: str = 'direct',
    ):
        self.model = model
        self.scale = scale
        self.guidance_type = guidance_type
        self.action_dim = action_dim
        self.model_z = model_z
        self.model_v = model_v

    def schedule_fn(self, t):
        #return t
        #return 1-t
        return 0.5 * (1 + torch.cos(t * math.pi))
        #return (torch.exp(-x) - math.exp(-1)) / (1 - math.exp(-1))
    
    

    def apply_guidance(self, xt, vt, grad_v, cond, t, values, eps=1e-8):
        """
        Apply the Cov-G method to compute a guided vector field.
        
        Args:
            xt: current state (B, horizon, transition_dim)
            vt: vector field predicted by the model (B, horizon, transition_dim)
            cond: conditions [(time, state), ...]
            t: current time (B,)
            values: value function outputs (B, 1)
            
        Returns:
            guided_vt: guided vector field (B, horizon, transition_dim)
        """

        # reward weighting approach (eq:guidance_matching_loss_g_4)
        guided_vt = vt +  grad_v * self.scale * self.schedule_fn(t)

        return guided_vt
        
    def _compute_z(self, x, cond, t):
        """
        Compute Z values.
        """
        if self.model_z is None:
            # use default
            return torch.ones(x.shape[0], device=x.device)
        else:
            # compute Z using the provided model
            z_pred = self.model_z(x, cond, t)  # (B, horizon, 1)
            return z_pred.squeeze(-1)[:, -1].exp().clamp(min=1e-8)  # (B,)

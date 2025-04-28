import torch
import torch.nn as nn
import math


class GuidanceMatcher:
    """
    Cov-G 방법을 구현하는 클래스
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
        Cov-G 방법을 적용하여 가이딩된 벡터 필드를 계산하는 메서드
        
        Args:
            xt: 현재 상태 (B, horizon, transition_dim)
            vt: 모델이 예측한 벡터 필드 (B, horizon, transition_dim)
            cond: 조건 [(time, state), ...]
            t: 현재 시간 (B,)
            values: 가치 함수 값 (B, 1)
            
        Returns:
            guided_vt: 가이딩된 벡터 필드 (B, horizon, transition_dim)
        """

        # 리워드 가중치 방법 (eq:guidance_matching_loss_g_4)
        guided_vt = vt +  grad_v * self.scale * self.schedule_fn(t)

        return guided_vt
        
    def _compute_z(self, x, cond, t):
        """
        Z 값을 계산하는 메서드
        """
        if self.model_z is None:
            # 기본값 사용
            return torch.ones(x.shape[0], device=x.device)
        else:
            # 모델을 사용하여 Z 계산
            z_pred = self.model_z(x, cond, t)  # (B, horizon, 1)
            return z_pred.squeeze(-1)[:, -1].exp().clamp(min=1e-8)  # (B,)

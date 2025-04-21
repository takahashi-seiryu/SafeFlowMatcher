import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
import diffuser.utils as utils
import pdb
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # CFM setting
        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)


        # Settings for compatibility with diffusion models (Not important for CFM)
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def conditioned_ode_func_record(self, t, x, cond, trajectory_list):
        """
        ODE vector field function with conditioning applied at each step.
        
        t (float or tensor): Current time step in the ODE solver.
        x (tensor): Current state.
        cond (dict): Conditioning dictionary used to fix specific time steps or values.
        trajectory_list (list): List to accumulate the trajectory.
        
        vt (tensor): Vector field output from the model.
        """
        trajectory_list.append(x)  # Append the current state to the trajectory list

        # 1. Apply conditioning to the current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute the vector field from the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)
        
        return vt

    @torch.no_grad()
    def conditioned_ode_func(self, t, x, cond):
        """
        Computes the ODE vector field with conditioning at each step
        """
        # 1. Apply condition to current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute vector field on the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)
        
        return vt

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        """
        Generate samples by solving the conditional ODE
        """
        # Initial noise
        x0 = torch.randn(shape).to(self.device)
        
        # Apply condition to initial state
        x0 = apply_conditioning(x0, cond, self.action_dim)
        
        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if return_diffusion:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond)

        # Solve ODE using wrapper
        traj = torchdiffeq.odeint(
            ode_fn,
            x0,
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        
        x1 = traj[-1]
        # Apply condition again at the end (for safety)
        x1 = apply_conditioning(x1, cond, self.action_dim)
        
        # pdb.set_trace()

        if return_diffusion:
            trajectory_list.append(x1) # append last step x
            return x1, torch.stack(trajectory_list, dim=1)
        return x1

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, return_diffusion=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        #horizon = 124
        #cond[767] = cond[383].clone()
        #cond.pop(383, None)
      
        #cond[horizon-1] = cond[383].clone()
        #cond.pop(383, None)

        #cond[179] = torch.tensor([[-0.3, 0, -0.0026,  0.0122]], device='cuda:0')
        #cond[383] = torch.tensor([[-0.8380, 0.7506, -0.0026,  0.0122]], device='cuda:0')
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_diffusion=return_diffusion, *args, **kwargs)

    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device
    
    #------------------------------------------ training ------------------------------------------#
    
    def loss(self, x, cond):
        x = x.to(self.device)
        batch_size = len(x)

        t = torch.rand(batch_size, device=x.device)
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)

        # Generate xt and flow field ut at time t
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply condition
        xt = apply_conditioning(xt, cond, self.action_dim)
        
        # Compute vector field
        vt = self.model(xt, None, t) # if there are cond, modify None -> cond

        # Compute loss
        loss, info = self.loss_fn(vt, ut)
        
        return loss, info
    
    def violation_forecasting(self, cond, *args, **kwargs): #CBF wise로 받게
        #1. init) x0 = [c, c,c,c,c,....,c]
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        x0 = torch.randn(shape).to(self.device)
        # CBF 중심 좌표 추출 (첫 번째 장애물 사용)
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        pos_y = torch.full((batch_size, horizon), off_y, device=self.device)
        pos_x = torch.full((batch_size, horizon), off_x, device=self.device)
        x0[:, :, self.action_dim] = pos_y
        x0[:, :, self.action_dim + 1] = pos_x

        #2. [c, c,c,c,c,....,c] -> [start, c,c,c,c,....,end]
        x0 = apply_conditioning(x0, cond, self.action_dim)

        #3. v0 구하기
        t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
        v0 = self.model(x0, None, t_batch) 
        
        # 4. 닫힌 형식으로 r0(CBF 경계까지의 거리) 계산=====이거뭐야 무서워...
        # 위치 및 속도 성분 추출
        pos_y = x0[:, :, self.action_dim]      # 위치 y 성분 (batch_size, horizon)
        pos_x = x0[:, :, self.action_dim + 1]  # 위치 x 성분 (batch_size, horizon)
        v_y = v0[:, :, self.action_dim]        # 속도 y 성분 (batch_size, horizon)
        v_x = v0[:, :, self.action_dim + 1]    # 속도 x 성분 (batch_size, horizon)
        # CBF1 매개변수 (타원형 장애물)
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # 2차 방정식 계수: A*r^2 + B*r + C = 0
        A = (v_y/yr)**2 + (v_x/xr)**2
        B = 2*(pos_y - off_y)*v_y/yr**2 + 2*(pos_x - off_x)*v_x/xr**2
        C = ((pos_y - off_y)/yr)**2 + ((pos_x - off_x)/xr)**2 - (1 + 0.01)
        # 판별식 계산
        discriminant = B**2 - 4*A*C
        # r0를 기본값 1로 초기화
        r0 = torch.ones_like(pos_y)
        # 유효한 교차점이 있는 경우 (판별식 >= 0)
        valid_mask = discriminant >= 0
        if valid_mask.any():
            # 두 가능한 거리 계산
            r1 = (-B[valid_mask] + torch.sqrt(discriminant[valid_mask])) / (2*A[valid_mask])
            r2 = (-B[valid_mask] - torch.sqrt(discriminant[valid_mask])) / (2*A[valid_mask])
            # 가장 작은 양수 거리 찾기
            r_stack = torch.stack([r1, r2], dim=1)
            r_stack[r_stack <= 0] = float('inf')  # 음수 값을 무한대로 설정
            r_min, _ = torch.min(r_stack, dim=1)
            # 양수 해가 없으면 기본값 r0=1 유지
            r_min[r_min == float('inf')] = 1.0
            # 유효한 점들의 r0 업데이트
            r0[valid_mask] = r_min

        #5. x'0와 x1 (one-shot) 구하기
        x0_prime = x0.clone()
        x0_prime[:, :, self.action_dim:self.action_dim+2] = x0[:, :, self.action_dim:self.action_dim+2] + r0.unsqueeze(-1) * v0[:, :, self.action_dim:self.action_dim+2]
        x1 = x0.clone()
        x1[:, :, self.action_dim:self.action_dim+2] = x0[:, :, self.action_dim:self.action_dim+2] + v0[:, :, self.action_dim:self.action_dim+2]

        #6. v'0 구하기
        v0_prime = self.model(x0_prime, None, t_batch)
        
        #7. v0 * v'0 값 구하기 & [0,1]로 step = f ============================================================
        v0_pos = v0[:, :, self.action_dim:self.action_dim+2]
        v0_prime_pos = v0_prime[:, :, self.action_dim:self.action_dim+2]
            # 내적 계산 (normalize해서 cos 값 구하기)
        v0_norm = torch.norm(v0_pos, dim=2, keepdim=True)
        v0_prime_norm = torch.norm(v0_prime_pos, dim=2, keepdim=True)
        cosine_sim = torch.sum(v0_pos * v0_prime_pos, dim=2) / (v0_norm.squeeze(-1) * v0_prime_norm.squeeze(-1) + 1e-8)
        # #print(cosine_sim)
            # 위반 점수 계산 (1 - cosine_sim로 방향 변화 측정)
        # cosine_bin 계산: cosine_sim > 0이면 1, 아니면 0 할당
        cosine_bin = (cosine_sim > 0).float()
        # 위치 성분(pos_y, pos_x)만 추출하여 거리 계산
        distance = torch.norm( # CBF 예상위치(x1)와 가장 먼 위치(x0_prime) 찾기
            x1[:, :, self.action_dim:self.action_dim*2] - 
            x0_prime[:, :, self.action_dim:self.action_dim*2], 
            dim=2
        )  # (batch_size, horizon)
        # 수정된 violation_score 계산
        #violation_score = (1 - cosine_bin) * distance

                # x1의 위치 좌표 추출
        x1_pos_y = x1[:, :, self.action_dim]      # 위치 y 성분 (batch_size, horizon)
        x1_pos_x = x1[:, :, self.action_dim + 1]  # 위치 x 성분 (batch_size, horizon)

        # CBF 값 계산 (타원형 장애물)
        cbf_value = ((x1_pos_y - off_y)/yr)**2 + ((x1_pos_x - off_x)/xr)**2 - 1 - 0.01

        # CBF 위반 여부 확인 (음수면 위반)
        cbf_violation = (cbf_value < 0).float()

        violation_score = cbf_violation * (1 - cosine_bin) * distance
        # =================================================================================================

        #8. 가장 심각한 위반 지점 찾기
        batch_indices = torch.arange(batch_size, device=self.device)
        time_indices = torch.argmax(violation_score, dim=1)
        
        #9. sub_goals = (t, traj[t])
        for i in range(batch_size):
            t_idx = time_indices[i].item()
            sub_goals = x0_prime[i, t_idx, self.action_dim:].unsqueeze(0)

            # 시각화 함수 호출
        visualize_cbf_violation(x0, x0_prime, x1, v0, cbf_value, cosine_sim, t_idx, self.action_dim, "analy_1")

        t_idx = (((t_idx+2)//4)*4) # conv 구조가 stride떄문에 4의 배수 horizon만 받음... 따라서 t에 오차가 생기는데 추후 수정필
        return t_idx, sub_goals
    
    def violation_forecasting2(self, cond, *args, **kwargs): #CBF wise로 받게
        #1. init) x0 = [c, c,c,c,c,....,c]
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        x0 = torch.randn(shape).to(self.device)
        
        # CBF 중심 좌표 추출 (두 번째 장애물 사용) - 수정된 부분
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        pos_y = torch.full((batch_size, horizon), off_y, device=self.device)
        pos_x = torch.full((batch_size, horizon), off_x, device=self.device)
        x0[:, :, self.action_dim] = pos_y
        x0[:, :, self.action_dim + 1] = pos_x

        #2. [c, c,c,c,c,....,c] -> [start, c,c,c,c,....,end]
        x0 = apply_conditioning(x0, cond, self.action_dim)

        #3. v0 구하기
        t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
        v0 = self.model(x0, None, t_batch) 
        
        # 4. 4제곱 슈퍼 타원에 대한 CBF 경계까지의 거리 계산
        # 위치 및 속도 성분 추출
        # 위치 및 속도 성분 추출
        pos_y = x0[:, :, self.action_dim]      # 위치 y 성분 (batch_size, horizon)
        pos_x = x0[:, :, self.action_dim + 1]  # 위치 x 성분 (batch_size, horizon)
        v_y = v0[:, :, self.action_dim]        # 속도 y 성분 (batch_size, horizon)
        v_x = v0[:, :, self.action_dim + 1]    # 속도 x 성분 (batch_size, horizon)
        
        # CBF 매개변수 (4차 슈퍼타원)
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        
        # r0 초기값 설정 (1.0으로 시작)
        r0 = torch.ones_like(pos_y)
        
        # 뉴턴-랩슨 방법을 벡터화하여 구현
        max_iters = 5  # 반복 횟수 (일반적으로 3-5회면 충분히 수렴)
        r = r0.clone()
        
        for _ in range(max_iters):
            # 현재 r 값에서의 위치 계산
            y_r = pos_y + r * v_y
            x_r = pos_x + r * v_x
            
            # CBF 함수값 계산: ((y-off_y)/yr)^4 + ((x-off_x)/xr)^4 - 1 - 0.01
            cbf_val = ((y_r - off_y)/yr)**4 + ((x_r - off_x)/xr)**4 - 1 - 0.01
            
            # CBF 함수의 r에 대한 미분값 계산
            # d/dr[((y-off_y)/yr)^4 + ((x-off_x)/xr)^4]
            # = 4*((y-off_y)/yr)^3 * v_y/yr + 4*((x-off_x)/xr)^3 * v_x/xr
            cbf_deriv = 4*((y_r - off_y)/yr)**3 * v_y/yr + 4*((x_r - off_x)/xr)**3 * v_x/xr
            
            # 미분값이 너무 작은 경우 발산 방지
            valid_deriv = (cbf_deriv.abs() > 1e-8)
            
            # 뉴턴-랩슨 업데이트: r = r - f(r)/f'(r)
            update = torch.zeros_like(r)
            update[valid_deriv] = cbf_val[valid_deriv] / cbf_deriv[valid_deriv]
            r = r - update
            
            # r 값이 음수가 되지 않도록 보정
            r = torch.clamp(r, min=0.1)
        
            # 최종 r0 값 설정
            r0 = r
        
        #5. x'0와 x1 (one-shot) 구하기
        x0_prime = x0.clone()
        x0_prime[:, :, self.action_dim:self.action_dim+2] = x0[:, :, self.action_dim:self.action_dim+2] + r0.unsqueeze(-1) * v0[:, :, self.action_dim:self.action_dim+2]
        x1 = x0.clone()
        x1[:, :, self.action_dim:self.action_dim+2] = x0[:, :, self.action_dim:self.action_dim+2] + v0[:, :, self.action_dim:self.action_dim+2]

        # 위치 성분 간 거리 계산
        distance = torch.norm(
            x1[:, :, self.action_dim:self.action_dim*2] - 
            x0_prime[:, :, self.action_dim:self.action_dim*2], 
            dim=2
        )  # (batch_size, horizon)

        # x1의 위치 좌표 추출
        x1_pos_y = x1[:, :, self.action_dim]
        x1_pos_x = x1[:, :, self.action_dim + 1]

        # 4제곱 슈퍼 타원 CBF 값 계산
        cbf_value = ((x1_pos_y - off_y)/yr)**4 + ((x1_pos_x - off_x)/xr)**4 - 1 - 0.01

        # CBF 위반 여부 확인 (음수면 위반)
        cbf_violation = (cbf_value < 0).float()

        #================== andgle noise -- oneshot noise둘다 사용
        v0_prime = self.model(x0_prime, None, t_batch)
        v0_pos = v0[:, :, self.action_dim:self.action_dim+2]
        v0_prime_pos = v0_prime[:, :, self.action_dim:self.action_dim+2]
        v0_norm = torch.norm(v0_pos, dim=2, keepdim=True)
        v0_prime_norm = torch.norm(v0_prime_pos, dim=2, keepdim=True)
        cosine_sim = torch.sum(v0_pos * v0_prime_pos, dim=2) / (v0_norm.squeeze(-1) * v0_prime_norm.squeeze(-1) + 1e-8)
        cosine_bin = (cosine_sim > -0.25).float()
        #==================

        # 위반 점수 계산
        violation_score = cbf_violation * (1 - cosine_bin) * distance

        #8. 가장 심각한 위반 지점 찾기
        batch_indices = torch.arange(batch_size, device=self.device)
        time_indices = torch.argmax(violation_score, dim=1)
        
        #9. sub_goals = (t, traj[t])
        for i in range(batch_size):
            t_idx = time_indices[i].item()
            sub_goals = x0_prime[i, t_idx, self.action_dim:].unsqueeze(0)

        visualize_cbf_violation(x0, x0_prime, x1, v0, cbf_value, cosine_sim, t_idx, self.action_dim, "analy_2")

        t_idx = (((t_idx+2)//4)*4) # conv 구조가 stride떄문에 4의 배수 horizon만 받음... 따라서 t에 오차가 생기는데 추후 수정필
        return t_idx, sub_goals

    

    def forward(self, cond, *args, **kwargs):
        # 1. CBF violation 예측) 일단 함수안에 CBF넣어둠
        # 2. sub goal로 job 나누기
        # 3. 이어붙이기
        t, sub_goal = self.violation_forecasting(cond,  *args, **kwargs)
        t2, sub_goal2 = self.violation_forecasting2(cond,  *args, **kwargs)
        print(t2, t)
        cond1 = cond.copy()
        cond1[t2-1] = sub_goal2
        cond1.pop(self.horizon-1,None)

        cond2 = cond.copy()
        cond2[0] = sub_goal2
        cond2[(t-1)-t2] = sub_goal
        cond2.pop(self.horizon-1,None)

        cond3 = cond.copy()
        cond3[0] = sub_goal
        cond3[(self.horizon-1)-t] = cond3[self.horizon-1].clone()
        cond3.pop(self.horizon-1,None)

        print("cond1: ", cond1)
        print("step: ", t2)
        print("cond2: ", cond2)
        print("step: ", t-t2)
        print("cond3: ", cond3)
        print("step: ", self.horizon-t)

        print("traj1 go")
        x1_1, traj_1 = self.conditional_sample(cond=cond1, *args, horizon=t2, **kwargs)
        print("traj2 go")
        x1_2, traj_2 = self.conditional_sample(cond=cond2, *args, horizon=t-t2, **kwargs)
        print("traj3 go")
        x1_3, traj_3 = self.conditional_sample(cond=cond3, *args, horizon=self.horizon-t, **kwargs)

        # 궤적 시각화 함수 호출
        visualize_trajectory(x1_1, x1_2, x1_3, self.action_dim,
                            title="CBF 기반 경로 계획 궤적",
                            save_path="trajectory_segments.png")

        # concat x1 & traj
        #x1 = torch.cat([x1_1[:, :-1], x1_2], dim=1)        
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        #traj = torch.cat([traj_1[:, :-1], traj_2], dim=1)
        traj = torch.cat([traj_1, traj_2, traj_3], dim=2)
        
        return x1, traj
        #return x1_1, traj_1

    def forward_orig(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
    

# =========== under is func for visualization =========

def visualize_cbf_violation(x0, x0_prime, x1, v0, cbf_vi, cosine_sim, t_idx, action_dim, name):
    # 배치 차원이 있으면 첫 번째 배치만 사용
    if x0.dim() > 2:
        batch_idx = 0
        x0 = x0[batch_idx]
        x0_prime = x0_prime[batch_idx]
        x1 = x1[batch_idx]
        v0 = v0[batch_idx]
    
    # 위치 정보 추출 (pos_y, pos_x) - detach() 추가
    pos_x0 = x0[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x0_prime = x0_prime[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x1 = x1[:, action_dim:action_dim+2].detach().cpu().numpy()
    
    # 벡터 정보 추출 (v_y, v_x) - detach() 추가
    v0_pos = v0[:, action_dim:action_dim+2].detach().cpu().numpy()
    
    # 1. 궤적 시각화 (x0, x0_prime)
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    # 궤적 그리기
    plt.plot(pos_x0[:, 1], pos_x0[:, 0], 'b-', label='x0 trajectory')
    plt.plot(pos_x0_prime[:, 1], pos_x0_prime[:, 0], 'g-', label='x0_prime trajectory')
    plt.plot(pos_x1[:, 1], pos_x1[:, 0], 'r-', label='x1 trajectory')
    # t_idx 강조
    plt.scatter(pos_x0[t_idx, 1], pos_x0[t_idx, 0], color='red', s=100, label=f'x0 at t={t_idx}')
    plt.scatter(pos_x0_prime[t_idx, 1], pos_x0_prime[t_idx, 0], color='purple', s=100, label=f'x0_prime at t={t_idx}')
    plt.scatter(pos_x1[t_idx, 1], pos_x1[t_idx, 0], color='black', s=100, label=f'x0 at t={t_idx}')
    # 두 점 연결
    plt.plot([pos_x0[t_idx, 1], pos_x0_prime[t_idx, 1]], 
             [pos_x0[t_idx, 0], pos_x0_prime[t_idx, 0]], 
             'r--', label='Connection at t_idx')
    plt.title('Trajectories of x0 and x0_prime')
    plt.xlabel('pos_x')
    plt.ylabel('pos_y')
    plt.legend()
    plt.grid(True)
    # x축과 y축 범위를 [-1, 1]로 설정
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # 축 비율을 동일하게 설정 (정사각형 플롯 보장)
    plt.gca().set_aspect('equal')
    # Y축 반전 추가
    plt.gca().invert_yaxis()
    
    # 2. CBF_value 그래프 - detach() 추가
    plt.subplot(3, 1, 2)
    time_steps = np.arange(len(cbf_vi[0]))
    plt.plot(time_steps, cbf_vi[0].detach().cpu().numpy(), 'b-', label='CBF_valye')
    #pdb.set_trace()
    # t_idx 강조
    plt.scatter(t_idx, cbf_vi[0][t_idx].detach().cpu().numpy(), color='red', s=100)
    
    # cosine_sim = 0 선 추가
    plt.axhline(y=0, color='r', linestyle='--', label='cosine_sim = 0')
    
    plt.title('CBF value of x1')
    plt.xlabel('Time Step')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    # 3. cosine_sim 그래프 - detach() 추가
    plt.subplot(3, 1, 3)
    time_steps = np.arange(len(cosine_sim[0]))
    plt.plot(time_steps, cosine_sim[0].detach().cpu().numpy(), 'b-', label='cosine_sim')

    # t_idx 강조
    plt.scatter(t_idx, cosine_sim[0][t_idx].detach().cpu().numpy(), color='red', s=100)
    
    # cosine_sim = 0 선 추가
    plt.axhline(y=0, color='r', linestyle='--', label='cosine_sim = 0')
    
    plt.title('Cosine Similarity between v0 and v0_prime')
    plt.xlabel('Time Step')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(name + 'trajectory_and_CBF_and_cosine.png')
    plt.close()
    
    print(f"시각화 완료: {name + 'trajectory_and_CBF_and_cosine.png'}")
    return 'trajectory_and_CBF_and_cosine.png'

def visualize_trajectory(x1_1, x1_2, x1_3, action_dim, title="trajectory Visualization", save_path="trajectory_visualization.png"):
    """
    위치 좌표를 사용해 궤적을 시각화하는 함수
    
    Parameters:
    - x1_1, x1_2, x1_3: 궤적 세그먼트 텐서
    - action_dim: 위치 차원이 시작되는 인덱스
    - title: 플롯 제목
    - save_path: 시각화 저장 경로
    """
    import matplotlib.pyplot as plt
    
    # 각 세그먼트의 위치 데이터 추출 (batch_size=1 가정)
    # y와 x 위치 추출 (시각화를 위해 detach하고 numpy로 변환)
    pos_y_1 = x1_1[0, :, action_dim].detach().cpu().numpy()
    pos_x_1 = x1_1[0, :, action_dim+1].detach().cpu().numpy()
    
    pos_y_2 = x1_2[0, :, action_dim].detach().cpu().numpy()
    pos_x_2 = x1_2[0, :, action_dim+1].detach().cpu().numpy()
    
    pos_y_3 = x1_3[0, :, action_dim].detach().cpu().numpy()
    pos_x_3 = x1_3[0, :, action_dim+1].detach().cpu().numpy()
    
    # 플롯 생성
    plt.figure(figsize=(10, 8))
    
    # 각 세그먼트를 다른 색상으로 플롯
    plt.plot(pos_x_1, pos_y_1, 'b-', linewidth=2, label='1st segment')
    plt.plot(pos_x_2, pos_y_2, 'g-', linewidth=2, label='2st segment')
    plt.plot(pos_x_3, pos_y_3, 'r-', linewidth=2, label='3st segment')
    
    # 시작점과 끝점 표시
    plt.scatter(pos_x_1[0], pos_y_1[0], color='blue', s=100, marker='o', label='start point')
    plt.scatter(pos_x_3[-1], pos_y_3[-1], color='red', s=100, marker='o', label='end point')
    
    # 전환점 표시
    plt.scatter(pos_x_1[-1], pos_y_1[-1], color='purple', s=150, marker='*', label='seg_1')
    plt.scatter(pos_x_2[-1], pos_y_2[-1], color='purple', s=150, marker='*', label='seg_2')
    
    # 레이블과 제목 추가
    plt.xlabel('위치 X')
    plt.ylabel('위치 Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # x축과 y축 범위를 [-1, 1]로 설정
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    # 축 비율을 동일하게 설정 (정사각형 플롯 보장)
    plt.gca().set_aspect('equal')
    
    # Y축 반전 추가
    plt.gca().invert_yaxis()
    
    # 이미지 저장
    plt.savefig(save_path)
    
    # 이미지 저장
    plt.savefig(save_path)
    plt.close()
    
    print(f"궤적 시각화가 {save_path}에 저장되었습니다")
    return save_path

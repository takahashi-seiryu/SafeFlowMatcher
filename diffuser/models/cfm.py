import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from diffuser.models import cbf
from diffuser.models.temporal_film import TemporalFiLM
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
        hidden_dim=128,  # Added for temporal film
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.hidden_dim = hidden_dim

        # Temporal FiLM layer for guidance
        feature_dim = horizon * (observation_dim + action_dim)  # Total flattened dimension
        self.temporal_film = TemporalFiLM(feature_dim)

        # CFM setting
        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(self._guided_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # Safety
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        # Settings for compatibility with diffusion models
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

    def _guided_model(self, t, x):
        """
        Wrapper for the model that applies temporal guidance through FiLM
        Args:
            t: time steps [batch_size]
            x: input states [batch_size, horizon, transition_dim]
        Returns:
            vt: velocity prediction [batch_size, horizon, transition_dim]
        """
        batch_size = x.shape[0]
        
        x_flat = x.reshape(batch_size, -1)  # [batch_size, horizon * transition_dim]
        x_modulated = self.temporal_film(x_flat, t)  # [batch_size, horizon * transition_dim]
        x_reshaped = x_modulated.reshape(x.shape)  # [batch_size, horizon, transition_dim]
        
        vt = self.model(x_reshaped, None, t)  # [batch_size, horizon, transition_dim]
        return vt

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
        vt = self._guided_model(t_batch, x_cond)
        
        # if self.safety_enabled and self.cbf is not None:
        #     # Flatten: x, vt are [B, H, D]
        #     B, H, D = vt.shape
        #     corrected = torch.zeros_like(vt)

        #     for h in range(H):
        #         x_cur  = x[:, h:h+1, :]               # [B,1,D]
        #         x_next = x_cur + vt[:, h:h+1, :]      # naive next-step

        #         x_corr, safe_vals = self.cbf.apply(x_cur, x_next, t=t)
        #         corrected[:, h:h+1, :] = x_corr - x_cur  # corrected vt

        #         # (Optional) log safety metrics
        #         # print(f"[t={t:.3f}] h={h}: min safety = {[v.item() for v in safe_vals]}")

        #     return corrected  # [B, H, D]
    
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
        vt = self._guided_model(t_batch, x_cond)

        # if self.safety_enabled and self.cbf is not None:
        #     # Flatten: x, vt are [B, H, D]
        #     B, H, D = vt.shape
        #     corrected = torch.zeros_like(vt)

        #     for h in range(H):
        #         x_cur  = x[:, h:h+1, :]               # [B,1,D]
        #         x_next = x_cur + vt[:, h:h+1, :]      # naive next-step

        #         x_corr, safe_vals = self.cbf.apply(x_cur, x_next, t=t)
        #         corrected[:, h:h+1, :] = x_corr - x_cur  # corrected vt

        #         # (Optional) log safety metrics
        #         # print(f"[t={t:.3f}] h={h}: min safety = {[v.item() for v in safe_vals]}")

        #     return corrected  # [B, H, D]
        
        return vt

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, record_traj=False):
        """
        Generate samples by solving the conditional ODE
        """

        # ================ one-shot initialization ================
        batch_size = len(cond[0])
        x0 = torch.randn(shape).to(self.device)
        x0 = apply_conditioning(x0, cond, self.action_dim)
        
        #3. v0 구하기
        t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
        v0 = self.model(x0, None, t_batch) 

        #4. x1_pred 구하기
        x1_pred = x0.clone()
        x1_pred = x0 + v0
        # ================ one-shot initialization ================
        x1_pred = apply_conditioning(x1_pred, cond, self.action_dim)
        
        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if record_traj:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond)

        # Solve ODE using wrapper
        traj = torchdiffeq.odeint(
            ode_fn,
            x1_pred,
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        
        x1 = traj[-1]
        # Apply condition again at the end (for safety)
        x1 = apply_conditioning(x1, cond, self.action_dim)
        
        # pdb.set_trace()
        if record_traj:
            trajectory_list.append(x1) # append last step x
            return x1, torch.stack(trajectory_list, dim=1)
        return x1
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond, verbose=True, record_traj=False):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        # ================ one-shot initialization ================
        batch_size = len(cond[0])
        x0 = torch.randn(shape).to(self.device)
        x0 = apply_conditioning(x0, cond, self.action_dim)
        
        #3. v0 구하기
        t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
        v0 = self.model(x0, None, t_batch) 

        #4. x1_pred 구하기
        x1_pred = x0.clone()
        x1_pred = x0 + v0
        # ================ one-shot initialization ================
        x1_pred = apply_conditioning(x1_pred, cond, self.action_dim)

        T = self.n_timesteps + 1
        time = torch.linspace(0, 1, T).to(self.device)
        # traj = [x0]
        traj = [x1_pred]

        for i in range(1, T):
            #print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time[i-1]
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * (1. / self.n_timesteps)
                x_corr, _ = self.cbf.apply(x_now, x_next_naive, t=t_now) 
                dx = x_corr - x_now
            else:
                dx = u_raw * (1. / self.n_timesteps)

            x_next = x_now + dx 
            #print(f"dx : {dx}")
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,256,:,:], traj_tensor  # sample, diffusion_paths
        else:
            return traj_tensor[:,256,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, record_traj=True, return_diffusion=False, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        if self.safety_enabled: # Planning
            samples = self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs)
        else: # Training
            samples = self.p_sample_loop(shape, cond, record_traj=record_traj, *args, **kwargs)
            
        if return_diffusion:
            return samples, None  # CFM doesn't have diffusion trajectory
        return samples

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
        vt = self._guided_model(t, xt)

        # Compute loss
        loss, info = self.loss_fn(vt, ut)
        
        return loss, info
    
    def violation_forecasting(self, x0, x1, n=2, cbf_pos_y=2, cbf_pos_x=5, cbf_r_y=2, cbf_r_x=2): #CBF wise로 받게
        # CBF 중심 좌표 추출 (두 번째 장애물 사용) - 수정된 부분
        off_y = 2*(cbf_pos_y-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2*(cbf_pos_x-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        pos_y = torch.full((1, self.horizon), off_y, device=self.device)
        pos_x = torch.full((1, self.horizon), off_x, device=self.device)

        # 기존 v0 대신 장애물 중심에서 x1으로 향하는 벡터 계산
        # 로봇 위치 추출
        x1_pos_y = x1[:, :, self.action_dim]
        x1_pos_x = x1[:, :, self.action_dim + 1]

        # 장애물 중심에서 로봇 위치로 향하는 벡터 계산
        v_y = x1_pos_y - pos_y  # 로봇 y 위치 - 장애물 y 중심
        v_x = x1_pos_x - pos_x  # 로봇 x 위치 - 장애물 x 중심

        # 새로운 방향 벡터를 원래 v0의 형태로 복사
        v = x1.clone()
        v[:, :, self.action_dim] = v_y    # 속도 y 성분
        v[:, :, self.action_dim + 1] = v_x  # 속도 x 성분
        
        # 4. 2제곱 슈퍼 타원에 대한 CBF 경계까지의 거리 계산
        # 위치 및 속도 성분 추출
        # 위치 및 속도 성분 추출
        v_y = v[:, :, self.action_dim]        # 속도 y 성분 (batch_size, horizon)
        v_x = v[:, :, self.action_dim + 1]    # 속도 x 성분 (batch_size, horizon)
        
        # CBF 매개변수 (2차 슈퍼타원)
        yr = cbf_r_y*1/(self.norm_maxs[0] - self.norm_mins[0])
        xr = cbf_r_x*1/(self.norm_maxs[1] - self.norm_mins[1])
    
        denominator = ((v_y/yr)**n + (v_x/xr)**n)**(1/n)
        r0 = (1 + 1e-2)**(1/n) / denominator
        #r0 = torch.clamp(r0, min=0.1, max=10.0)

        #5. x'0와 x1 (one-shot) 구하기
        x0_prime = x1.clone()
        x0_prime[:, :, self.action_dim] = pos_y
        x0_prime[:, :, self.action_dim + 1] = pos_x
        x0_prime[:, :, self.action_dim:self.action_dim+2] = x0_prime[:, :, self.action_dim:self.action_dim+2] + r0.unsqueeze(-1) * v[:, :, self.action_dim:self.action_dim+2]

        # x1의 위치 좌표 추출
        x1_pos_y = x1[:, :, self.action_dim]
        x1_pos_x = x1[:, :, self.action_dim + 1]

        # 2제곱 슈퍼 타원 CBF 값 계산
        cbf_value = ((x1_pos_y - off_y)/yr)**n + ((x1_pos_x - off_x)/xr)**n - 1 - 0.01

        #8. 가장 심각한 위반 지점 찾기
        time_indices = torch.argmin(cbf_value, dim=1)
        t_idx = time_indices[0].item()

        if t_idx != 0:
            #9. find sub_goal's velocity
            y_velocity = (1/xr)**n * (x0_prime[0, t_idx, self.action_dim+1] - off_x)**(n-1)
            x_velocity = -(1/yr)**n * (x0_prime[0, t_idx, self.action_dim] - off_y)**(n-1)
            p_velocity = torch.stack([y_velocity, x_velocity])
            p_velocity = p_velocity / torch.norm(p_velocity, dim=0)
            size_of_v = torch.norm(x1[0, t_idx, 2*self.action_dim:])
            # dot product of p_velocity and v0
            # 시간 순서를 고려한 위반 구간 찾기
            violation_indices = torch.where(cbf_value[0] < 0)[0]
            entry_idx = violation_indices.min()  # 첫 위반 지점
            exit_idx = violation_indices.max()   # 마지막 위반 지점
            
            in_point = x1[0, entry_idx, self.action_dim:2*self.action_dim]
            out_point = x1[0, exit_idx, self.action_dim:2*self.action_dim]
            in_out_v = out_point - in_point
            cosine_sim = torch.dot(p_velocity, in_out_v)
            if cosine_sim < 0:
                sign = -1
            else:
                sign = 1
            x0_prime[0, t_idx, 2*self.action_dim:] = p_velocity * size_of_v * sign 


        #9. sub_goals = (t, traj[t])
        sub_goals = x0_prime[0, t_idx, self.action_dim:].unsqueeze(0)

        #visualize_cbf_violation_cos(x0, x0_prime, x1, v0, cbf_value, cosine_sim, t_idx, self.action_dim, "analy_2")
        visualize_cbf_violation(x0, x0_prime, x1, cbf_value, t_idx, self.action_dim, f"analy_{n}")

        t_idx = (((t_idx+2)//4)*4) # conv 구조가 stride떄문에 4의 배수 horizon만 받음... 따라서 t에 오차가 생기는데 추후 수정필
        return t_idx, sub_goals

    def forward(self, cond, *args, **kwargs):
        #1. init) x0 [start, noise, noise, noise, noise, ...., end]
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        x0 = torch.randn(shape).to(self.device)
        x0 = apply_conditioning(x0, cond, self.action_dim)
        
        #3. v0 구하기
        t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
        v0 = self.model(x0, None, t_batch) 

        #4. x1_pred 구하기
        x1_pred = x0.clone()
        x1_pred = x0 + v0
        #====================================================== # 
        # todo)
        # 1. now is using velocitys as 0 but use x1_pred  <-  vertical to contact line
        # 2. make CBF as Class
        #====================================================== # 


        t, sub_goal = self.violation_forecasting(x0, x1_pred, n=2, cbf_pos_y=5, cbf_pos_x=5.8)
        t2, sub_goal2 = self.violation_forecasting(x0, x1_pred, n=4, cbf_pos_y=2, cbf_pos_x=5.3)

        sub_goal_list = []
        sub_goal_list.append([0, cond[0]])
        sub_goal_list.append([self.horizon, cond[self.horizon-1]])
        if t != 0:
            sub_goal_list.append([t, sub_goal])
        if t2 != 0:
            sub_goal_list.append([t2, sub_goal2])
        sub_goal_list = sorted(sub_goal_list, key=lambda x: x[0])

        cond_list = []
        step_list = []
        for i in range(len(sub_goal_list)-1):
            step_temp = sub_goal_list[i+1][0] - sub_goal_list[i][0]
            cond_temp = {}

            cond_temp[0] = sub_goal_list[i][1]

            cond_temp[step_temp -1] = sub_goal_list[i+1][1]
            cond_list.append(cond_temp)
            step_list.append(step_temp)

        x1_list = []
        traj_list = []
        for i in range(len(cond_list)):
            print(f"task {i}/ step: {step_list[i]}, cond: {cond_list[i]}")
            x1_temp, traj_temp = self.conditional_sample(cond=cond_list[i], *args, horizon=step_list[i], **kwargs)
            x1_list.append(x1_temp)
            traj_list.append(traj_temp)

        # 궤적 시각화 함수 호출
        visualize_trajectory(x1_list, self.action_dim,
                            title="CBF-based trajectory planning",
                            save_path="trajectory_segments.png")

        # concat x1 & traj   
        x1 = torch.cat(x1_list, dim=1)
        traj = torch.cat(traj_list, dim=2)
        print(x1.shape)
        
        return x1, traj

# =========== under is func for visualization ============
def visualize_cbf_violation(x0, x0_prime, x1, cbf_vi, t_idx, action_dim, name):
    # 배치 차원이 있으면 첫 번째 배치만 사용
    if x0.dim() > 2:
        batch_idx = 0
        x0 = x0[batch_idx]
        x0_prime = x0_prime[batch_idx]
        x1 = x1[batch_idx]
    
    # 위치 정보 추출 (pos_y, pos_x) - detach() 추가
    pos_x0 = x0[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x0_prime = x0_prime[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x1 = x1[:, action_dim:action_dim+2].detach().cpu().numpy()
    
        # 기존 figure 설정 대신 사용
    plt.figure(figsize=(15, 8))  # 전체 figure 크기 조정

    # GridSpec으로 레이아웃 설정
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])

    # 1. 궤적 시각화 (왼쪽 전체 영역 사용)
    ax1 = plt.subplot(gs[:, 0])  # 왼쪽 열 전체 사용
    # 기존 궤적 그리기 코드...
    #plt.plot(pos_x0[:, 1], pos_x0[:, 0], 'b-', label='x0 trajectory')
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
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal')  # 정사각형 비율 유지
    plt.gca().invert_yaxis()

    # 2. CBF_value 그래프 (오른쪽 상단)
    ax2 = plt.subplot(gs[0, 1])
    time_steps = np.arange(len(cbf_vi[0]))
    plt.plot(time_steps, cbf_vi[0].detach().cpu().numpy(), 'b-', label='CBF_valye')
    plt.scatter(t_idx, cbf_vi[0][t_idx].detach().cpu().numpy(), color='red', s=100)
    plt.axhline(y=0, color='r', linestyle='--', label='cosine_sim = 0')
    plt.yscale('symlog', linthresh=1.1)
    plt.title('CBF value of x1')
    plt.xlabel('Time Step')
    plt.ylabel('CBF value')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1000)

    # 3. cosine_sim 그래프 (오른쪽 하단)
    ax3 = plt.subplot(gs[1, 1])
    time_steps = np.arange(len(cbf_vi[0]))
    plt.plot(time_steps, cbf_vi[0].detach().cpu().numpy(), 'b-', label='cosine_sim')
    plt.scatter(t_idx, cbf_vi[0][t_idx].detach().cpu().numpy(), color='red', s=100)
    plt.axhline(y=0, color='r', linestyle='--', label='cosine_sim = 0')
    plt.title('Cosine Similarity between v0 and v0_prime')
    plt.xlabel('Time Step')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(name + 'trajectory_and_CBF.png')
    plt.close()

def visualize_trajectory(x1_list, action_dim, title="trajectory Visualization", save_path="trajectory_visualization.png"):
    """
    위치 좌표를 사용해 궤적을 시각화하는 함수
    
    Parameters:
    - x1_1, x1_2, x1_3: 궤적 세그먼트 텐서
    - action_dim: 위치 차원이 시작되는 인덱스
    - title: 플롯 제목
    - save_path: 시각화 저장 경로
    """
    # 플롯 생성
    plt.figure(figsize=(10, 8))

    num_x1 = len(x1_list)
    if num_x1 >0:
        x1_1 = x1_list[0]
        pos_y_1 = x1_1[0, :, action_dim].detach().cpu().numpy()
        pos_x_1 = x1_1[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_1, pos_y_1, 'b-', linewidth=2, label='1st segment')
    if num_x1 >1:
        x1_2 = x1_list[1]
        pos_y_2 = x1_2[0, :, action_dim].detach().cpu().numpy()
        pos_x_2 = x1_2[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_2, pos_y_2, 'g-', linewidth=2, label='2st segment')
    if num_x1 >2:
        x1_3 = x1_list[2]
        pos_y_3 = x1_3[0, :, action_dim].detach().cpu().numpy()
        pos_x_3 = x1_3[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_3, pos_y_3, 'r-', linewidth=2, label='3st segment')
    
    # 시작점과 끝점 표시
    plt.scatter(pos_x_1[0], pos_y_1[0], color='blue', s=100, marker='o', label='start point')
    if num_x1 == 3:
        plt.scatter(pos_x_3[-1], pos_y_3[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='red', s=100, marker='o', label='end point')
    
    # 전환점 표시
    if num_x1 > 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='purple', s=150, marker='*', label='seg_1')
    if num_x1 > 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='purple', s=150, marker='*', label='seg_2')
    
    # 레이블과 제목 추가
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
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
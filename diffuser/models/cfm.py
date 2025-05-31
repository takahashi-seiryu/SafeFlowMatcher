import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torch.distributions.normal import Normal
import diffuser.utils as utils
import pdb
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

# For NLL
class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # expects model(x, cond, t)

    def forward(self, x, t):
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        return self.model(x, None, t_batch)

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

        # One-shot initialization
        self.one_shot_enabled = False

        # Safety
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = []
        self.safe2 = []

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
    def p_sample_loop(self, shape, cond, verbose=True, record_traj=False):
        """
        Generate samples by solving the conditional ODE
        """
        # Initial noise
        x0 = torch.randn(shape).to(self.device)
        
        # Apply condition to initial state
        x0 = apply_conditioning(x0, cond, self.action_dim)

        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if record_traj:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond)
        
        iter_start = time.time()
        # Solve ODE using wrapper
        traj = torchdiffeq.odeint(
            ode_fn,
            x0,
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        iter_end = time.time()
        iter_time = iter_end - iter_start

        x1 = traj[-1]
        # Apply condition again at the end (for safety)
        x1 = apply_conditioning(x1, cond, self.action_dim)
        
        # pdb.set_trace()
        if record_traj:
            trajectory_list.append(x1) # append last step x
            return x1, torch.stack(trajectory_list, dim=1), [iter_time/self.n_timesteps]
        return x1
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond, verbose=True, record_traj=False):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        # ================ Model Loading ================
        _bs = len(cond[0])
        _x = torch.randn(shape).to(self.device)
        _t = torch.zeros((_bs,), device=self.device)
        _ = self.model(_x, None, _t) #model loading@@@@@@@
        OSI_start = time.time()
        # ================ one-shot initialization ================
        if self.one_shot_enabled:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            # Obtain velocity field for one-shot
            t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
            v0 = self.model(x0_1st_phase, None, t_batch) 
            # Obtain one-shot prediction (1-step Euler)
            x1_pred = x0_1st_phase.clone()
            x1_pred = x0_1st_phase + v0
            
            x0_2nd_phase = x1_pred
        # ================ Multi-step Planning ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)
        OSI_end = time.time()
        OSI_time = OSI_end - OSI_start


        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)
        n_timesteps = 256 # self.n_timesteps <------ fix this
        T = n_timesteps + 1
        # Uniform scheduling: lmbd = 0.0
        # Adaptive scheduling: lmbd = 1.0
        # time_list = self.adaptive_scheduling(T)
        time_list = torch.linspace(0, 1, T).to(self.device)  # uniform scheduling
        traj = [x0_2nd_phase]
        
        safe1, safe2 = [], []
        iter_time = 0
        for t in range(1, T*4 -3):
            i = t % 256 + 1
            iter_start = time.time()
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time_list[i-1]
            dt = time_list[i] - t_now
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            lmbd = 1.0  # bigger give more power to previous step (0,1)
            step = 2*((2*lmbd)/(n_timesteps *(n_timesteps + 1))) * (n_timesteps-(i-1))+ (1-lmbd)/(n_timesteps)
            #if i == T-1:
            #    step = (1/n_timesteps) * (n_timesteps-(i-1))
            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * step
                x_corr, safe_val = self.cbf.apply(x_now, x_next_naive, t=t_now)
                dx = x_corr - x_now
                iter_end = time.time()
                safe1.append(safe_val[0].unsqueeze(0))
                safe2.append(safe_val[1].unsqueeze(0))
            else:
                # dx = u_raw * dt
                dx = u_raw * step
                iter_end = time.time()

            x_next = x_now + dx
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_time += (iter_end - iter_start)
        
        #self.safe1 = torch.cat(safe1, dim=0)
        #self.safe2 = torch.cat(safe2, dim=0)

        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,1024,:,:], traj_tensor, [iter_time/n_timesteps, OSI_time]  # sample, diffusion_paths, [mean_iter_time, OSI_time]
            #return traj_tensor[:,n_timesteps,:,:], traj_tensor, [iter_time/n_timesteps, OSI_time]  # sample, diffusion_paths, [mean_iter_time, OSI_time]
        else:
            return traj_tensor[:,n_timesteps,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, record_traj=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        # if self.safety_enabled: # Planning
        if True: # Planning
            return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs)
        else: # Training
            return self.p_sample_loop(shape, cond, record_traj=record_traj, *args, **kwargs)

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

    # wthout segementing
    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

    # with segmenting
    # def forward(self, cond, *args, **kwargs):
    #     batch_size = len(cond[0])
    #     horizon = self.horizon
    #     shape = (batch_size, horizon, self.transition_dim)
        
    #     # 1. Initialize noisy trajectory with boundary condition
    #     x0 = torch.randn(shape).to(self.device)
    #     x0 = apply_conditioning(x0, cond, self.action_dim)
        
    #     # 2. Predict velocity
    #     t_batch = torch.zeros((batch_size,), device=self.device)
    #     v0 = self.model(x0, None, t_batch) 

    #     # 3. Predict trajectory via 1-step Euler
    #     x1_pred = x0.clone()
    #     x1_pred = x0 + v0
        
    #     # 4. Forecast violation
    #     t_list, sub_goal_list = self.cbf.forecast_violation(x0, x1_pred)
        
    #     # 5. Add start & end goals
    #     sub_goal_pairs = [[0, cond[0]]]
    #     for t, g in zip(t_list, sub_goal_list):
    #         if t != 0:
    #             sub_goal_pairs.append([t, g])
    #     sub_goal_pairs.append([horizon, cond[horizon - 1]])
    #     sub_goal_pairs = sorted(sub_goal_pairs, key=lambda x: x[0])

    #     # 6. Build condition sets per segment
    #     cond_list = []
    #     step_list = []
    #     for i in range(len(sub_goal_pairs) - 1):
    #         t0, g0 = sub_goal_pairs[i]
    #         t1, g1 = sub_goal_pairs[i + 1]
    #         steps = t1 - t0
    #         step_list.append(steps)
    #         cond_list.append({0: g0, steps - 1: g1})
        
    #     # 7. Plan each segment
    #     x1_list = []
    #     traj_list = []
    #     for i in range(len(cond_list)):
    #         print(f"task {i}/ step: {step_list[i]}, cond: {cond_list[i]}")
    #         x1_temp, traj_temp = self.conditional_sample(cond=cond_list[i], *args, horizon=step_list[i], **kwargs)
    #         x1_list.append(x1_temp)
    #         traj_list.append(traj_temp)

    #     visualize_trajectory(x1_list, self.action_dim,
    #                         title="CBF-based trajectory planning",
    #                         save_path="logs/trajectory_segments.png")

    #     x1 = torch.cat(x1_list, dim=1)
    #     traj = torch.cat(traj_list, dim=2)
        
    #     return x1, traj

    # ------------------------------------------ NLL calculation ------------------------------------------#
    # def compute_nll(self, x1, num_steps, exact_div=False):
    #     device = self.device
    #     x1 = x1.to(device)
    #     B, H, D = x1.shape

    #     # prior log-probability
    #     def log_p0(x: torch.Tensor) -> torch.Tensor:
    #         return Normal(0.0, 1.0).log_prob(x).sum(dim=(1, 2))   # [B]

    #     # pre-sample Hutchinson Rademacher noise
    #     if not exact_div:
    #         z = (torch.randint_like(x1, low=0, high=2) * 2 - 1).to(device)  # ±1

    #     # ODE system
    #     def dynamics(t, states):
    #         x_t, log_det = states                                # xt:[B,H,D] , log_det:[B]
    #         x_t = x_t.requires_grad_(True)

    #         # velocity field v_t(x)
    #         t_batch = torch.full((B,), t.item(), device=device)
    #         ut = self.model(x_t, None, t_batch)                  # [B,H,D]

    #         # divergence -- exact or Hutchinson
    #         if exact_div:
    #             div = torch.zeros(B, device=device)
    #             # trace of Jacobian
    #             for idx in range(ut.flatten(1).shape[1]):
    #                 div += torch.autograd.grad(
    #                     ut.flatten(1)[:, idx].sum(), x_t,
    #                     create_graph=False, retain_graph=True
    #                 )[0].flatten(1)[:, idx]
    #         else:
    #             dot = (ut * z).sum()                            # scalar
    #             grad = torch.autograd.grad(dot, x_t, create_graph=False, retain_graph=True)[0]
    #             div = (grad * z).flatten(1).sum(dim=1)          # [B]

    #         return (ut, div)                                    # x_dot = u , l_dot = div(v_t)

    #     # integrate backward  t:1 -> 0
    #     t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # uniform scheduling
    #     #t_grid = self.adaptive_scheduling(num_steps)  # adaptive scheduling
    #     #t_grid = torch.flip(t_grid, dims=[0])  # reverse time grid 0->1
    #     y0 = (x1, torch.zeros(B, device=device))

    #     sol_x, sol_log = torchdiffeq.odeint(
    #         dynamics,
    #         y0,
    #         t_grid,
    #         method='euler',  # 'euler', 'rk4', 'dopri5'
    #         atol=1e-5,
    #         rtol=1e-5
    #     )

    #     x0      = sol_x[-1]                  # latent sample at t = 0
    #     log_det = sol_log[-1]                # ∫ div(v_t) dt   (sign handled by time grid)

    #     # final log-prob & NLL
    #     log_px1 = log_p0(x0) + log_det       # log p(x1)
    #     nll     = -log_px1.mean()            # averaged over batch

    #     return x0, nll
    
    def compute_nll(self, x1, num_steps, exact_div=False):
        """
        Computes the Negative Log-Likelihood (NLL) via instantaneous change of variables.
        Integrates the ODE backward from t=1 to t=0 using torchdiffeq.odeint.
        """
        device = self.device
        x1 = x1.to(device)
        B, H, D = x1.shape

        # Prior log-probability p₀ (standard Gaussian)
        def log_p0(x: torch.Tensor) -> torch.Tensor:
            return Normal(0.0, 1.0).log_prob(x).sum(dim=(1, 2))  # [B]

        # Pre-sample Hutchinson ±1 noise on device if needed
        if not exact_div:
            z = (torch.randint_like(x1, low=0, high=2, device=device) * 2 - 1).float()

        # ODE RHS: dx/dt = u(x, t), d(log_det)/dt = div u(x, t)
        def dynamics(t, states):
            x_t, log_det = states              # x_t: [B,H,D], log_det: [B]
            x_t = x_t.requires_grad_(True)

            # compute velocity field at time t
            t_batch = torch.full((B,), t, device=device)
            ut = self.model(x_t, None, t_batch)  # [B,H,D]

            # divergence term
            if exact_div:
                div = torch.zeros(B, device=device)
                flat_ut = ut.flatten(1)         # [B, H*D]
                for idx in range(flat_ut.shape[1]):
                    # sum_i ∂ut_i/∂x_t_i
                    g = torch.autograd.grad(
                        flat_ut[:, idx].sum(), x_t,
                        retain_graph=True, create_graph=False
                    )[0]
                    div += g.flatten(1)[:, idx]
            else:
                # Hutchinson estimator per sample
                # dot: [B] = ∑_i ut_i * z_i
                dot = (ut * z).flatten(1).sum(dim=1)
                # sum over batch to get scalar for autograd
                dot_sum = dot.sum()
                grad = torch.autograd.grad(
                    dot_sum, x_t,
                    retain_graph=True, create_graph=False
                )[0]
                # divergence per sample: zᵀ (∂(zᵀu)/∂x)
                div = (grad * z).flatten(1).sum(dim=1)  # [B]

            return ut, div

        # integrate backward: t=1 → 0
        t_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        y0 = (x1, torch.zeros(B, device=device))

        sol_x, sol_log = torchdiffeq.odeint(
            dynamics, y0, t_grid,
            method='euler', atol=1e-5, rtol=1e-5
        )

        x0 = sol_x[-1]               # latent at t=0
        log_det = sol_log[-1]        # ∫ div dt

        # log p(x1) = log p₀(x0) + ∫ div
        log_px1 = log_p0(x0) + log_det
        nll = -log_px1.mean()        # average NLL

        return x0, nll
        
    def adaptive_scheduling(self, num_steps):
            """
            Return a 1-D tensor of length ``num_steps+1`` that goes from 0 → 1.
            Early Δt's are large (grid is sparse) and later Δt's are small (dense).

            The k-th step (1-based) is
                Δt_k = 2 (N-k+1) / (N·(N+1))
            so that Δt_k ∝ (1 − t_k)   and   ΣΔt_k = 1.
            """
            N = num_steps
            # k = N, N-1, …, 1  →  Δt_N is the smallest
            k = torch.arange(N, 0, -1, dtype=torch.float32, device=self.device)  # shape (N,)
            steps = 2.0 * k / (N * (N + 1))                                 # shape (N,)

            # build t-grid : [0, Δt₁, Δt₁+Δt₂, …, 1]
            t_grid = torch.cat(
                (torch.zeros(1, dtype=torch.float32, device=self.device),
                torch.cumsum(steps, dim=0))
            )
            return t_grid

# =========== under is func for visualization ============
def visualize_trajectory(x1_list, action_dim, title="trajectory Visualization", save_path="trajectory_visualization.png"):
    """
    Function to visualize trajectories using position coordinates
    
    Parameters:
    - x1_list: List of trajectory segment tensors
    - action_dim: Index where position dimensions start
    - title: Plot title
    - save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))

    num_x1 = len(x1_list)
    if num_x1 > 0:
        x1_1 = x1_list[0]
        pos_y_1 = x1_1[0, :, action_dim].detach().cpu().numpy()
        pos_x_1 = x1_1[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_1, pos_y_1, 'b-', linewidth=2, label='1st segment')
    if num_x1 > 1:
        x1_2 = x1_list[1]
        pos_y_2 = x1_2[0, :, action_dim].detach().cpu().numpy()
        pos_x_2 = x1_2[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_2, pos_y_2, 'g-', linewidth=2, label='2nd segment')
    if num_x1 > 2:
        x1_3 = x1_list[2]
        pos_y_3 = x1_3[0, :, action_dim].detach().cpu().numpy()
        pos_x_3 = x1_3[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_3, pos_y_3, 'r-', linewidth=2, label='3rd segment')
    
    # Mark start and end points
    plt.scatter(pos_x_1[0], pos_y_1[0], color='blue', s=100, marker='o', label='start point')
    if num_x1 == 3:
        plt.scatter(pos_x_3[-1], pos_y_3[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='red', s=100, marker='o', label='end point')
    
    # Mark transition points
    if num_x1 > 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='purple', s=150, marker='*', label='seg_1')
    if num_x1 > 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='purple', s=150, marker='*', label='seg_2')
    
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"Trajectory visualization saved at {save_path}")
    return save_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from diffuser.models import cbf
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

        # One-shot initialization
        self.one_shot_enabled = False

        # Safety
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

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
        # ================ N_step Planning ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = self.n_timesteps + 1
        time = torch.linspace(0, 1, T).to(self.device)
        traj = [x0_2nd_phase]

        for i in range(1, T):
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
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
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,256,:,:], traj_tensor  # sample, diffusion_paths
        else:
            return traj_tensor[:,256,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, record_traj=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        if self.safety_enabled: # Planning
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

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

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
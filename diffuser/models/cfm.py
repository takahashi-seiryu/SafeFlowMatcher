import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
# from diffuser.models import cbf
from diffuser.sampling.guides import ValueGuide
from diffuser.models.guidance_matcher import GuidanceMatcher
import diffuser.utils as utils
import pdb
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        hidden_dim=128,  # Added for temporal film
        
    ):
        super().__init__()
        self.mean = 0  # for normalization
        self.std = 0
        self.min = 0  
        self.max = 0
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.hidden_dim = hidden_dim


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

        # reward guidance attributes
        self.guidance_enabled = False
        self.guidance_matcher = None
        self.value_guide = None
        self.guidance_scale = 1.0

        # only for diffusion
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
        vt = self._guided_model(t_batch, x_cond)

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
        
        return vt

    def enable_guidance(self, value_model, guidance_type='direct', scale=1.0):
        """
        Enable reward guidance.
        
        Args:
            value_model: value function model
            guidance_type: guidance method ('direct', 'use_learned_v', 'rw')
            scale: guidance scale
        """
        
        self.guidance_enabled = True
        self.value_guide = ValueGuide(value_model)
        self.guidance_matcher = GuidanceMatcher(
            model=self.model,
            action_dim=self.action_dim,
            scale=scale,
            guidance_type=guidance_type
        )
        self.guidance_scale = scale

    def _guided_model(self, t, x):
        """
        Model call with guidance applied.
        """
        # compute the base vector field
        vt = self.model(x, None, t)  # [batch_size, horizon, transition_dim]
        # apply reward guidance
        if self.guidance_enabled and self.value_guide is not None:
            # compute value function
            x = x.detach().requires_grad_()
            with torch.enable_grad():
                x1_pred = x + (1-t)*vt
                values, grad_v = self.value_guide.gradients(x1_pred, None, t)
            
            # apply guidancefrom qpth.qp import QPFunction, QPSolvers
            vt = self.guidance_matcher.apply_guidance(x, vt, grad_v, None, t, values.unsqueeze(-1))
        
        return vt

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, record_traj=False, **kwargs):
        """
        Generate samples by solving the conditional ODE
        """
        # Initial noise
        x0 = torch.randn(shape).to(self.device)
        
        # Apply condition to initial state
        x0 = apply_conditioning(x0, cond, self.action_dim)
        #pdb.set_trace()
        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if record_traj:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list, **kwargs)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond, **kwargs)

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
            return x1, torch.stack(trajectory_list, dim=1), 0
        return x1
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond, verbose=True, record_traj=False, **kwargs):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        # ================ one-shot initialization ================
        # if self.one_shot_enabled:
        prediction = True
        if prediction:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            # Obtain velocity field for one-shot
            t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
            v0 = self._guided_model(t_batch, x0_1st_phase)
            # v0 = self.model(x0_1st_phase, None, t_batch) 

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
        z =  2*(self.n_timesteps+1) / self.n_timesteps
        for i in range(1, T):
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time[i-1]
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self._guided_model(t_batch, x_now)
           
            dt = 1/ self.n_timesteps
            if prediction:
                one_minus_t = (self.n_timesteps - (i-1))/(self.n_timesteps)
                dt = z*one_minus_t*dt
            dx = u_raw * dt

            x_next = x_now + dx
            
            b_min = torch.tensor(0.0, device=x_now.device) # only using When no CBF
            ##########################################walker2d
            # x_next, b_min = self.invariance(x_now, x_next)  # RoS diffuser
            # x_next, b_min = self.invariance_cf(x_now, x_next)   #RoS diffuser, closed form

            ##########################################hopper
            # x_next, b_min = self.invariance_hopper(x_now, x_next)  # RoS diffuser
            # x_next, b_min = self.invariance_hopper_cf(x_now, x_next)   #RoS diffuser, closed form
            
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,self.n_timesteps,:,:], traj_tensor, b_min #b_min  # sample, diffusion_paths
        else:
            return traj_tensor[:,self.n_timesteps,:,:], b_min #b_min               # just sample

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
            return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, **kwargs)
        else: # Training
            return self.p_sample_loop(shape, cond, record_traj=record_traj, **kwargs)


    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device
    
    ###################################################################walker2d  
    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):    # RoS diffuser

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        # height = (height - self.mean[0]) / self.std[0]
        height =  2 * (height - mins[0]) / (maxs[0] - mins[0]) - 1

        #CBF
        ############################################ceiling
        epsilon = 1
        rho = 0.99
        
        b = height - x[:,6:7] # - 0.1*x[:,15:16]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        h = Lfb + epsilon * torch.sign(b) * torch.abs(b)**rho
   
        q = -torch.cat([ref[:,6:7]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # RoS diffuser closed-form

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        # height = (height - self.mean[0]) / self.std[0]
        height =  2 * (height - mins[0]) / (maxs[0] - mins[0]) - 1

        #CBF
        ############################################ceiling
        epsilon = 1
        rho = 0.99
        
        b0 = height - x[:,6:7] # - 0.1*x[:,15:16]    # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        h0 = Lfb + epsilon * torch.sign(b0) * torch.abs(b0)**rho

        b1 = x[:,6:7] + 10
        Lgbu1 = 1*torch.ones_like(x[:,6:7])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + epsilon * torch.sign(b1) * torch.abs(b1)**rho

        q = -torch.cat([ref[:,6:7]], dim = 1).to(G0.device)  #, ref[:,15:16]

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b0)  # + 0.01  # for robustness

###################################################################hopper  
    @torch.no_grad()   #only for sampling
    def invariance_hopper(self, x, xp1):   # RoS diffuser (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        # height = (height - self.mean[0]) / self.std[0]
        height =  2 * (height - mins[0]) / (maxs[0] - mins[0]) - 1

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cf(self, x, xp1):  # RoS diffuser closed form (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        # height = (height - self.mean[0]) / self.std[0]
        height =  2 * (height - self.min[0]) / (self.max[0] - self.min[0]) - 1

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]  # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,3:4])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + k*(x[:,3:4] + 10)
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G0.device)  #, ref[:,15:16]

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b)  # + 0.01  # for robustness
    
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

class ValueCFM(CFM):

    def loss(self, x, *args):
        return self.p_losses(x, *args)
    
    def p_losses(self, x_start, cond, target):
        x_0 = torch.randn_like(x_start)
        t, x_noisy, ut = self.FM.sample_location_and_conditional_flow(x_0, x_start)
        
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

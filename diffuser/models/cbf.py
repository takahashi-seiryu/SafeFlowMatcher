import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable
import pdb

class CBF:
    def __init__(self, norm_mins, norm_maxs, args):
        self.device = norm_mins.device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.horizon = args.horizon
        self.obstacles = args.obstacles
        self.cbf_solver = args.cbf_solver
        self.cbf_method = args.cbf_method
        self.action_dim = 2
        
        # Parameters for CBF
        self.alpha = args.eps
        self.rho = args.rho

        self.robust_term = args.robust_term
        self.relax_threshold = args.relax_threshold
        self.a = 100
        self.t_bias = 0.90

        # Precompute normalization factors
        self.xr = 2 / (self.norm_maxs[1] - self.norm_mins[1])
        self.yr = 2 / (self.norm_maxs[0] - self.norm_mins[0])
    
    @torch.no_grad()
    def compute_single_constraint(self, x, obs, t=None):
        cx, cy = obs['center']
        r = obs['radius']
        off_x = 2 * (cx - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2 * (cy - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        dx = (x[:,3:4] - off_x) / self.xr
        dy = (x[:,2:3] - off_y) / self.yr
        order = obs['order']

        # Lie derivative
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr

        # Finite-time parameter
        alpha = self.alpha
        rho = self.rho
        delta = self.robust_term

        # construct b(x)
        b = dy**order + dx**order - (1 + delta)#**order # we need to eliminate order squere

        if self.cbf_method == 'robust':
            G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)

            b = dy**order + dx**order - 1
            finite_time_term = torch.sign(b - delta) * torch.abs(b - delta)**rho
            h = alpha * finite_time_term

        elif self.cbf_method == 'relax':
            # sign = 30.0 if (t is not None and t <= self.relax_threshold) else 0.0
            if t <= self.relax_threshold:
                ratio = t / self.relax_threshold
                sign = 200.0 * (1 - math.exp(3 * (ratio - 1)))  # exp drop near t=relax_threshold
            else:
                sign = 0.0

            rx = sign * torch.ones_like(L1)
            G = torch.cat([-L1, -L2, rx], dim=1).unsqueeze(1)

            b = dy**order + dx**order - (1 + delta) **order # we need to eliminate order squere
            finite_time_term = torch.sign(b) * torch.abs(b)**rho
            h = alpha * finite_time_term

        elif self.cbf_method == 'time':
            G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)

            s = torch.sigmoid(self.a * (t - self.t_bias))
            ds = self.a * s * (1 - s)
            b = dy**order + dx**order - s
            finite_time_term = torch.sign(b - delta) * torch.abs(b - delta)**rho
            h = ds + alpha * finite_time_term

        else:
            raise ValueError(f"Unknown CBF method {self.cbf_method}")

        safe = torch.min(b)
        return G, h, safe
    
    @torch.no_grad()
    def solve_qp(self, u_ref, G, h, method='robust'):
        """
        min ‖u - u_ref‖²
        s.t. G u ≤ h
        """
        if method in ['robust', 'time']:
            q = -2 * u_ref[:, 2:4]      # [B, 2]: desired move -(Δx, Δy)

            Q = 2 * torch.eye(2, device=self.device).unsqueeze(0).expand(u_ref.size(0), 2, 2)
        elif method == 'relax':
            q_u = -u_ref[:, 2:4]      # [B, 2]: desired move -(Δx, Δy)
            q_r = torch.zeros_like(q_u[:, :1])  # [B, 2]
            q = 2 * torch.cat([q_u, q_r], dim=1)       # [B, 3]

            Q = 2 * torch.eye(3, device=self.device).unsqueeze(0).expand(u_ref.size(0), 3, 3)
        else:
            raise ValueError(f"Unknown method {method}")
                
        # No equality constraints
        e = torch.empty(0, device=self.device)

        out = QPFunction(
            eps=1e-12,                       # Tolerance (convergence criterion)
            verbose=-1,                      # Output level (-1: off, 0: summary, 1: detailed)
            notImprovedLim=10,                # Allowed number of iterations without improvement
            maxIter=20,                      # Maximum number of iterations
            solver=QPSolvers.PDIPM_BATCHED,  # Solver to use
            check_Q_spd=True                 # Whether to check if Q is SPD (Symmetric Positive Definite)
        )(Q, q, G, h, e, e)
        return out

    def solve_closed_form(self, u_ref, G, h, method='robust'):
        """
        Closed-form solution
        """
        if method in ['robust', 'time']:
            u_bar = u_ref[:, 2:4]   # [B, 2] desired move
        elif method == 'relax':
            u = u_ref[:, 2:4]   # [B, 2] desired move
            u_relax = torch.zeros_like(u[:, :1])
            u_bar = torch.cat([u, u_relax], dim=1)  # [B, 4]
        else:
            raise ValueError(f"Unknown method {method}")
        
        G0 = G[:, 0, :]         # [B, 2]
        G1 = G[:, 1, :]         # [B, 2]
        h0 = h[:, 0:1]          # [B, 1]
        h1 = h[:, 1:2]          # [B, 1]

        # inner products (y_i^T y_j)
        y1_bar = G0
        y2_bar = G1

        p1_bar = h0 - torch.sum(G0 * u_bar, dim=1, keepdim=True)
        p2_bar = h1 - torch.sum(G1 * u_bar, dim=1, keepdim=True)

        G_mat = torch.cat([
            torch.sum(y1_bar * y1_bar, dim=1, keepdim=True).unsqueeze(0),
            torch.sum(y1_bar * y2_bar, dim=1, keepdim=True).unsqueeze(0),
            torch.sum(y2_bar * y1_bar, dim=1, keepdim=True).unsqueeze(0),
            torch.sum(y2_bar * y2_bar, dim=1, keepdim=True).unsqueeze(0),
        ], dim=0)  # shape: [4, B, 1]

        # stability terms
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # compute λ1
        lambda1 = torch.where(
            G_mat[2] * w_p2_bar < G_mat[3] * p1_bar,
            torch.zeros_like(p1_bar),
            torch.where(
                G_mat[1] * w_p1_bar < G_mat[0] * p2_bar,
                w_p1_bar / G_mat[0],
                torch.clamp(
                    G_mat[3] * p1_bar - G_mat[2] * p2_bar,
                    max=0
                ) / (G_mat[0] * G_mat[3] - G_mat[1] * G_mat[2] + 1e-6)
            )
        )

        # compute λ2
        lambda2 = torch.where(
            G_mat[2] * w_p2_bar < G_mat[3] * p1_bar,
            w_p2_bar / G_mat[3],
            torch.where(
                G_mat[1] * w_p1_bar < G_mat[0] * p2_bar,
                torch.zeros_like(p1_bar),
                torch.clamp(
                    G_mat[0] * p2_bar - G_mat[1] * p1_bar,
                    max=0
                ) / (G_mat[0] * G_mat[3] - G_mat[1] * G_mat[2] + 1e-6)
            )
        )

        # u = u_ref + λ1 y1 + λ2 y2
        out = u_bar + lambda1 * y1_bar + lambda2 * y2_bar

        return out
    
    @torch.no_grad()
    def apply(self, x, xp1, t=None): # x = now, xp1 = next state
        # remove the leading batch‐of‐1 dim
        x   = x.squeeze(0)    # [B, state_dim]
        xp1 = xp1.squeeze(0)  # [B, state_dim]

        # desired increment
        ref = xp1 - x         # [B, state_dim]

        G_list, h_list, safe_vals = [], [], []
        for obs in self.obstacles:
            G_i, h_i, safe_i = self.compute_single_constraint(x, obs, t)
            G_list.append(G_i)
            h_list.append(h_i)
            safe_vals.append(safe_i)
        # if you have no obstacles, just apply the reference control
        if not G_list:
            out = ref[:,2:4]
        else:
            G = torch.cat(G_list, dim=1)  # [B, num_obs, dim]
            h = torch.cat(h_list, dim=1)  # [B, num_obs]

            if self.cbf_solver == 'qp':
                out = self.solve_qp(ref, G, h, method=self.cbf_method)
            elif self.cbf_solver == 'closed_form':
                out = self.solve_closed_form(ref, G, h, method=self.cbf_method)
            else:
                raise ValueError(f"Unknown CBF solver {self.cbf_solver}")
        # rebuild the next‐state
        rt = xp1.clone()
        rt[:,2:4] = x[:,2:4] + out[:, :2]
        return rt.unsqueeze(0), safe_vals
    
    @torch.no_grad()
    def apply_narrow(self, x, xp1, t):
        """
        Relaxed Safe Diffuser (ReS-diffuser) for maze2d-large-v1
        (narrow passage case)
        """
        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x
        if t <= self.relax_threshold:   # debug  10
            ratio = t / self.relax_threshold
            sign = 200.0 * (1 - math.exp(3 * (ratio - 1)))
            # sign = 1   # relax
        else:
            sign = 0   # non-relax

        # normalize obstacle 1,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])

        off_x = 2*(5.2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(1.8-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4  # 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        self.safe1 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0, rx0, rx0, rx0, rx0], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        ########################################### obs 2
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(4.8-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b2 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 #0.01
        Lfb = 0
        Lgbu12 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu22 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b2[:,0] + 0.01)

        G2 = torch.cat([-Lgbu12, -Lgbu22, rx0, rx1, rx0, rx0, rx0, rx0], dim = 1)
        G2 = G2.unsqueeze(1)
        k = 1
        h2 = Lfb + k*b2

        ########################################### obs 3
        off_x = 2*(2.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.3-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b3 = ((x[:,2:3] - off_y)/yr/0.5)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu13 = 4*((x[:,2:3] - off_y)/yr/0.5)**3/yr
        Lgbu23 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G3 = torch.cat([-Lgbu13, -Lgbu23, rx0, rx0, rx1, rx0, rx0, rx0], dim = 1)
        G3 = G3.unsqueeze(1)
        k = 1
        h3 = Lfb + k*b3

        ########################################### obs 4
        off_x = 2*(8.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(3.3-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b4 = ((x[:,2:3] - off_y)/yr/1.8)**4 + ((x[:,3:4] - off_x)/xr/1.8)**4 - 1 - 0.01
        Lfb = 0
        Lgbu14 = 4*((x[:,2:3] - off_y)/yr/1.8)**3/yr
        Lgbu24 = 4*((x[:,3:4] - off_x)/xr/1.8)**3/xr

        G4 = torch.cat([-Lgbu14, -Lgbu24, rx0, rx0, rx0, rx1, rx0, rx0], dim = 1)
        G4 = G4.unsqueeze(1)
        k = 1
        h4 = Lfb + k*b4

        ########################################### obs 5
        off_x = 2*(7.4-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(6.8-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b5 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4 #0.01
        Lfb = 0
        Lgbu15 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu25 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G5 = torch.cat([-Lgbu15, -Lgbu25, rx0, rx0, rx0, rx0, rx1, rx0], dim = 1)
        G5 = G5.unsqueeze(1)
        k = 1
        h5 = Lfb + k*b5

        ########################################### obs 6
        off_x = 2*(9.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(6.1-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        # CBF
        b6 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu16 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu26 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G6 = torch.cat([-Lgbu16, -Lgbu26, rx0, rx0, rx0, rx0, rx0, rx1], dim = 1)
        G6 = G6.unsqueeze(1)
        k = 1
        h6 = Lfb + k*b6

        b0 = torch.cat([b, b2, b3, b4, b5, b6], dim = 1)
        idx = torch.argmin(b0, dim = 1).cpu().numpy()
        G0 = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        h0 = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)
        rows = len(G0[:,0,0])
        G = []
        h = []
        for i in range(rows):
            G.append(G0[i:i+1,idx[i]:idx[i]+1])
            h.append(h0[i:i+1,idx[i]:idx[i]+1])
        G = torch.cat(G, dim = 0)
        h = torch.cat(h, dim = 0)

        # G = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        # h = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)

        # G = torch.cat([G1, G2, G3, G5], dim = 1)
        # h = torch.cat([h1, h2, h3, h5], dim = 1)
        
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0, q0, q0], dim = 1)
        Q = Variable(torch.eye(8))
        Q = Q.unsqueeze(0).expand(nBatch, 8, 8).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt, 0
        
    def calc_cbf(self, x):
        """
        Calculate the CBF value for a given state and obstacle.
        """
        cbf_values = []
        cbf_value = []
        for obs in self.obstacles:
            cx, cy = obs['center']
            off_x = 2 * (cx - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            off_y = 2 * (cy - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            dx = (x[:, :,3:4] - off_x) / self.xr
            dy = (x[:, :,2:3] - off_y) / self.yr
            order = obs['order']

            cbf_values.append(dy**order + dx**order - 1)
        for i in range(len(cbf_values[0][0])):
            cbf_value.append(min(cbf_values[0][0][i], cbf_values[1][0][i]))
            
        return cbf_value
    
    @torch.no_grad()
    def cbf_nv(self, x1):
        num = 0
        safe_l = []

        x1_pos_y = x1[:, :, self.action_dim]
        x1_pos_x = x1[:, :, self.action_dim + 1]

        for obs in self.obstacles:
            center = obs['center']
            n = obs['order']

            # Calculate the center
            off_y = 2*(center[1]-0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            off_x = 2*(center[0]-0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1


            # Calculate the CBF value
            cbf_value = ((x1_pos_y - off_y)/self.yr)**n + ((x1_pos_x - off_x)/self.xr)**n - 1 

            cbf_value = cbf_value.tolist()[0]
            safe_l.append(min(cbf_value))

        return safe_l
    
    @torch.no_grad()
    def calc_c_smooth(self, x1, eps_area: float = 1e-12, eps_len: float = 1e-12):
        X = x1
        if X.dim() == 2:        # [T, D] -> [1, T, D]
            X = X.unsqueeze(0)
        assert X.dim() == 3, "x1 must be [B, T, D] or [T, D]"

        B, T, D = X.shape
        if T < 3:
            # Return 0 when there are not enough segments
            cs = X.new_zeros((B,))
            return cs[0] if x1.dim() == 2 else cs

        # Extract coordinates (note: using y before x)
        y = X[:, :, self.action_dim]
        x = X[:, :, self.action_dim + 1]
        P = torch.stack([x, y], dim=-1)  # [B, T, 2]

        # p_{i-1}, p_i, p_{i+1}
        p_im1 = P[:, :-2, :]     # [B, T-2, 2]
        p_i   = P[:, 1:-1, :]
        p_ip1 = P[:, 2:, :]

        # Edge vectors
        v1 = p_i   - p_im1       # p_i   - p_{i-1}
        v2 = p_ip1 - p_i         # p_{i+1} - p_i
        v3 = p_ip1 - p_im1       # p_{i+1} - p_{i-1}

        # Edge lengths a,b,c
        a = torch.linalg.norm(v1, dim=-1)   # [B, T-2]
        b = torch.linalg.norm(v2, dim=-1)
        c = torch.linalg.norm(v3, dim=-1)

        # 2D cross product z-component (|u_x v_y - u_y v_x|)
        cross_z = v1[..., 0] * v3[..., 1] - v1[..., 1] * v3[..., 0]
        A = 0.5 * cross_z.abs()             # Triangle area

        # Curvature kappa = 4A/(a*b*c) with stabilization
        denom = (a * b * c).clamp(min=eps_len)
        kappa = (4.0 * A) / denom

        # Mask out invalid or near-linear segments
        mask = (a > eps_len) & (b > eps_len) & (c > eps_len) & (A > eps_area) & torch.isfinite(kappa)
        kappa = torch.where(mask, kappa, torch.zeros_like(kappa))

        # Batchwise mean
        curv_sum = kappa.sum(dim=1)                 
        counts   = mask.sum(dim=1)                  
        c_smooth = curv_sum / counts.clamp(min=1)   
        c_smooth = torch.where(counts > 0, c_smooth, torch.zeros_like(c_smooth))
        
        return c_smooth[0]
    
    @torch.no_grad()
    def calc_s_smooth(self, x1, dt: float = 1e-2, eps_dt: float = 1e-12, return_per_step: bool = False):
        X = x1
        if X.dim() == 2:  # [T, D] -> [1, T, D]
            X = X.unsqueeze(0)
        assert X.dim() == 3, "x1 must be [B, T, D] or [T, D]"

        B, T, D = X.shape
        if T < 3:
            out = X.new_zeros((B,))
            return out[0] if x1.dim() == 2 else out

        # Extract coordinates (using y/x indices; order irrelevant for norm)
        y = X[:, :, self.action_dim]
        x = X[:, :, self.action_dim + 1]
        P = torch.stack([x, y], dim=-1)  # [B, T, 2]

        # Second-order finite difference: p[i+2] - 2*p[i+1] + p[i]
        second_diff = P[:, 2:, :] - 2.0 * P[:, 1:-1, :] + P[:, :-2, :]  # [B, T-2, 2]

        # Stabilize dt
        dt2 = torch.as_tensor(dt, dtype=P.dtype, device=P.device)
        dt2 = (dt2 * dt2).clamp(min=eps_dt * eps_dt)

        # Acceleration vector and magnitude
        a_vec = second_diff / dt2  # [B, T-2, 2]
        a_mag = torch.linalg.norm(a_vec, dim=-1)  # [B, T-2]

        # Guard against NaN/Inf
        mask = torch.isfinite(a_mag)
        sums = torch.where(mask, a_mag, torch.zeros_like(a_mag)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        s_smooth = sums / counts

        return s_smooth[0]
        
        
    @torch.no_grad()
    def calc_smooth(self, x1):
        c_smooth = self.calc_c_smooth(x1)
        s_smooth = self.calc_s_smooth(x1)

        return c_smooth, s_smooth
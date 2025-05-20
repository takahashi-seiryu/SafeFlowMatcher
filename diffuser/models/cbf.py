import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qpth.qp import QPFunction, QPSolvers

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
        self.alpha = 0.5
        self.rho = 0.9

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
        off_x = 2 * (cx - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2 * (cy - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        dx = (x[:,3:4] - off_x) / self.xr
        dy = (x[:,2:3] - off_y) / self.yr
        order = obs['order']

        # b(x)
        b = dy**order + dx**order - 1 - self.robust_term

        # Lie derivative
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr

        # Finite-time parameter
        alpha = self.alpha
        rho = self.rho
        delta = self.robust_term

        if self.cbf_method == 'robust':
            G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)

            b = dy**order + dx**order - 1
            finite_time_term = torch.sign(b - delta) * torch.abs(b - delta)**rho
            h = alpha * finite_time_term

        elif self.cbf_method == 'relax':
            # sign = 30.0 if (t is not None and t <= self.relax_threshold) else 0.0

            if t <= self.relax_threshold:
                ratio = t / self.relax_threshold
                sign = 100.0 * (1 - math.exp(3 * (ratio - 1)))  # exp drop near t=relax_threshold
            else:
                sign = 0.0

            rx = sign * torch.ones_like(L1)
            G = torch.cat([-L1, -L2, rx], dim=1).unsqueeze(1)

            b = dy**order + dx**order - 1
            finite_time_term = torch.sign(b - delta) * torch.abs(b - delta)**rho
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

        safe = torch.min(b + delta)
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
            verbose=0,                       # Output level (-1: off, 0: summary, 1: detailed)
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
    def apply(self, x, xp1, t=None):
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
    def forecast_violation(self, x0, x1):
        """
        For each obstacle, predict the most severe CBF violation point 
        and generate corresponding sub-goals.

        returns:
            t_list: List[int] – times of violation per obstacle
            sub_goal_list: List[Tensor] – sub-goals per obstacle
        """
        t_list = []
        sub_goal_list = []

        x1_pos_y = x1[:, :, self.action_dim]
        x1_pos_x = x1[:, :, self.action_dim + 1]

        for obs in self.obstacles:
            center = obs['center']
            n = obs['order']

            # 1. Calculate the center
            off_y = 2*(center[1]-0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            off_x = 2*(center[0]-0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            pos_y = torch.full((1, self.horizon), off_y, device=self.device)
            pos_x = torch.full((1, self.horizon), off_x, device=self.device)

            # 2. Calculate the vector pointing from the obstacle center to the trajectory
            v_y = x1_pos_y - pos_y
            v_x = x1_pos_x - pos_x
            v = x1.clone()
            v[:, :, self.action_dim] = v_y
            v[:, :, self.action_dim + 1] = v_x

            # 3. Calculate the boundary distance r0
            denominator = ((v_y/self.yr)**n + (v_x/self.xr)**n)**(1/n)
            r0 = (1 + 1e-2)**(1/n) / denominator

            # 4. Generate x0' (push away from the obstacle center by a safe distance)
            x0_prime = x1.clone()
            x0_prime[:, :, self.action_dim] = pos_y
            x0_prime[:, :, self.action_dim + 1] = pos_x
            x0_prime[:, :, self.action_dim:self.action_dim+2] += \
            r0.unsqueeze(-1) * v[:, :, self.action_dim:self.action_dim+2]

            # 5. Calculate the CBF value
            cbf_value = ((x1_pos_y - off_y)/self.yr)**n + ((x1_pos_x - off_x)/self.xr)**n - 1 - 0.01

            # 6. Find the most severe violation time step
            t_idx = torch.argmin(cbf_value, dim=1)[0].item()

            # 7. If a violation actually occurred, adjust the velocity direction
            if t_idx != 0:
                y_velocity = (1/self.xr)**n * (x0_prime[0, t_idx, self.action_dim+1] - off_x)**(n-1)
                x_velocity = -(1/self.yr)**n * (x0_prime[0, t_idx, self.action_dim] - off_y)**(n-1)
                p_velocity = torch.stack([y_velocity, x_velocity])
                p_velocity /= torch.norm(p_velocity, dim=0)
                size_of_v = torch.norm(x1[0, t_idx, 2*self.action_dim:])

                violation_indices = torch.where(cbf_value[0] < 0)[0]
                entry_idx = violation_indices.min()
                exit_idx = violation_indices.max()
                in_point = x1[0, entry_idx, self.action_dim:2*self.action_dim]
                out_point = x1[0, exit_idx, self.action_dim:2*self.action_dim]
                in_out_v = out_point - in_point
                sign = -1 if torch.dot(p_velocity, in_out_v) < 0 else 1
                x0_prime[0, t_idx, 2*self.action_dim:] = p_velocity * size_of_v * sign

            sub_goal = x0_prime[0, t_idx, self.action_dim:].unsqueeze(0)

            visualize_cbf_violation(x0, x0_prime, x1, cbf_value, t_idx, 
                                    self.action_dim, f'obstacle_{center}_analy')

            # t_idx stride correction
            t_idx = (((t_idx+2)//4)*4)

            t_list.append(t_idx)
            sub_goal_list.append(sub_goal)

        return t_list, sub_goal_list
    
# =========== under is func for visualization ============
def visualize_cbf_violation(x0, x0_prime, x1, cbf_vi, t_idx, action_dim, name):
    # If there is a batch dimension, use only the first batch
    if x0.dim() > 2:
        batch_idx = 0
        x0 = x0[batch_idx]
        x0_prime = x0_prime[batch_idx]
        x1 = x1[batch_idx]
    
    # Extract position information (pos_y, pos_x) - added detach()
    pos_x0 = x0[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x0_prime = x0_prime[:, action_dim:action_dim+2].detach().cpu().numpy()
    pos_x1 = x1[:, action_dim:action_dim+2].detach().cpu().numpy()
    
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])

    # 1. Trajectory visualization (use the entire left area)
    ax1 = plt.subplot(gs[:, 0])
    # Existing trajectory drawing code...
    # plt.plot(pos_x0[:, 1], pos_x0[:, 0], 'b-', label='x0 trajectory')
    plt.plot(pos_x0_prime[:, 1], pos_x0_prime[:, 0], 'g-', label='x0_prime trajectory')
    plt.plot(pos_x1[:, 1], pos_x1[:, 0], 'r-', label='x1 trajectory')
    
    plt.scatter(pos_x0[t_idx, 1], pos_x0[t_idx, 0], color='red', s=100, label=f'x0 at t={t_idx}')
    plt.scatter(pos_x0_prime[t_idx, 1], pos_x0_prime[t_idx, 0], color='purple', s=100, label=f'x0_prime at t={t_idx}')
    plt.scatter(pos_x1[t_idx, 1], pos_x1[t_idx, 0], color='black', s=100, label=f'x0 at t={t_idx}')
    # Connect the two points
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
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()

    # 2. CBF_value graph
    ax2 = plt.subplot(gs[0, 1])
    time_steps = np.arange(len(cbf_vi[0]))
    plt.plot(time_steps, cbf_vi[0].detach().cpu().numpy(), 'b-', label='CBF_value')
    plt.scatter(t_idx, cbf_vi[0][t_idx].detach().cpu().numpy(), color='red', s=100)
    plt.axhline(y=0, color='r', linestyle='--', label='cosine_sim = 0')
    plt.yscale('symlog', linthresh=1.1)
    plt.title('CBF value of x1')
    plt.xlabel('Time Step')
    plt.ylabel('CBF value')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1000)

    # 3. Cosine similarity graph
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
    plt.savefig("logs/" + name + 'trajectory_and_CBF_and_cosine.png')
    plt.close()
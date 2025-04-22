import torch
from qpth.qp import QPFunction, QPSolvers

class CBF:
    def __init__(
        self, norm_mins, norm_maxs, obstacles,
        cbf_solver='qp', cbf_method='normal', robust_term=0.01, relax_threshold=0.9
    ):
        device = norm_mins.device  # make sure norms are already on device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = obstacles
        self.cbf_solver = cbf_solver
        self.robust_term = robust_term
        self.cbf_method = cbf_method
        self.relax_threshold = relax_threshold
        self.device = device

        # Parameters for CBF
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

        b = dy**order + dx**order - 1 - self.robust_term
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr
        Lfb = 0
        k = 1

        if self.cbf_method == 'robust':
            G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)

        elif self.cbf_method == 'relax':
            sign = 100.0 if (t is not None and t <= self.relax_threshold) else 0.0
            rx1 = sign * torch.ones_like(L1)
            rx0 = torch.zeros_like(L1)
            G = torch.cat([-L1, -L2, rx1, rx0], dim=1).unsqueeze(1)

        elif self.cbf_method == 'time':
            s = torch.sigmoid(self.a * (t - self.t_bias))
            Lfb = self.a * s * (1 - s)
            b = dy**order + dx**order - s - self.robust_term
            G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)

        else:
            raise ValueError(f"Unknown CBF method {self.cbf_method}")

        h = Lfb + k * b
        safe = torch.min(b + self.robust_term)
        return G, h, safe
    
    @torch.no_grad()
    def solve_qp(self, u_ref, G, h, method='robust'):
        """
        min ‖u - u_ref‖²
        s.t. G u ≤ h
        """
        if method in ['robust', 'time']:
            q = -u_ref[:, 2:4]      # [B, 2]: desired move -(Δx, Δy)

            Q = torch.eye(2, device=self.device).unsqueeze(0).expand(u_ref.size(0), 2, 2)
        elif method == 'relax':
            q_u = -u_ref[:, 2:4]      # [B, 2]: desired move -(Δx, Δy)
            q_r = torch.zeros_like(u_ref[:, 2:4])  # [B, 2]
            q = torch.cat([q_u, q_r], dim=1)       # [B, 4]

            Q = torch.eye(4, device=self.device).unsqueeze(0).expand(u_ref.size(0), 4, 4)
        else:
            raise ValueError(f"Unknown method {method}")
                
        # No equality constraints
        e = torch.empty(0, device=self.device)

        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
        return out

    def solve_closed_form(self, u_ref, G, h, method='robust'):
        """
        Closed-form solution
        """
        if method in ['robust', 'time']:
            u_bar = u_ref[:, 2:4]   # [B, 2] desired move
        elif method == 'relax':
            u = u_ref[:, 2:4]   # [B, 2] desired move
            u_relax = torch.zeros_like(u)
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
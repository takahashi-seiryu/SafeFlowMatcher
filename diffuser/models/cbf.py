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

        # Precompute normalization factors
        self.xr = 2 / (self.norm_maxs[1] - self.norm_mins[1])
        self.yr = 2 / (self.norm_maxs[0] - self.norm_mins[0])

    @torch.no_grad()
    def compute_cbf_circle(self, x, center, radius):
        # Normalize center just once clearly
        off_x = 2 * (center[0] - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2 * (center[1] - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1

        diff_y = (x[:, 2:3] - off_y) / self.yr
        diff_x = (x[:, 3:4] - off_x) / self.xr

        b = diff_y**2 + diff_x**2 - 1 - self.robust_term
        Lgbu1 = 2 * diff_y / self.yr
        Lgbu2 = 2 * diff_x / self.xr

        G = torch.stack([-Lgbu1, -Lgbu2], dim=-1)
        h = b
        safe = torch.min(b + self.robust_term)

        return G, h, safe

    @torch.no_grad()
    def compute_cbf_4th_order(self, x, center, radius):
        # Normalize center once clearly
        off_x = 2 * (center[0] - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2 * (center[1] - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1

        diff_y = (x[:, 2:3] - off_y) / self.yr
        diff_x = (x[:, 3:4] - off_x) / self.xr

        b = diff_y**4 + diff_x**4 - 1 - self.robust_term
        Lgbu1 = 4 * diff_y**3 / self.yr
        Lgbu2 = 4 * diff_x**3 / self.xr

        G = torch.stack([-Lgbu1, -Lgbu2], dim=-1)
        h = b
        safe = torch.min(b + self.robust_term)

        return G, h, safe
    
    @torch.no_grad()
    def solve_qp(self, ref, G, h):
        q = -ref[:, 2:4].to(self.device)
        Q = torch.eye(2, device=self.device).unsqueeze(0).expand(ref.size(0), -1, -1)
        e = torch.empty(0, device=self.device)
        return QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

    def solve_qp_relax(self, ref, G, h):
        """
        QP with relaxation variables r1, r2 (dim=4)
        min ‖u - u_ref‖²
        s.t. G [u; r] ≤ h
        """
        u_bar = ref[:, 2:4]                     # [B, 2]
        q_u = -u_bar                            # [B, 2]
        q_r = torch.zeros_like(q_u)             # [B, 2]
        q = torch.cat([q_u, q_r], dim=1)        # [B, 4]

        Q = torch.eye(4, device=self.device).unsqueeze(0).expand(ref.size(0), 4, 4)
        e = torch.empty(0, device=self.device)

        # G: [B, num_obs, 4], h: [B, num_obs]
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        return out  # includes relaxation; only u part will be used

    @torch.no_grad()
    def apply(self, x, xp1, t=None):
        # remove the leading batch‐of‐1 dim
        x   = x.squeeze(0)    # [B, state_dim]
        xp1 = xp1.squeeze(0)  # [B, state_dim]

        # desired increment
        ref = xp1 - x         # [B, state_dim]

        G_list, h_list, safe_vals = [], [], []

        for obs in self.obstacles:
            # normalize obstacle center once
            cx, cy = obs['center']
            off_x = 2*(cx - 0.5 - self.norm_mins[1])/(self.norm_maxs[1]-self.norm_mins[1]) - 1
            off_y = 2*(cy - 0.5 - self.norm_mins[0])/(self.norm_maxs[0]-self.norm_mins[0]) - 1

            # extract the normalized y,x coords (same as apply_test)
            dy = (x[:,2:3] - off_y)/self.yr   # [B,1]
            dx = (x[:,3:4] - off_x)/self.xr   # [B,1]

            if self.cbf_method == 'time':
                # Time-varying CBF (TVS)
                t_bias = self.relax_threshold
                t_bias = 0.90  # e.g., 0.5 if normalized t ∈ [0, 1]
                a = 1
                s = torch.sigmoid(a*(t - t_bias))  # scalar
                Lfb = a * s * (1 - s)              # time-varying Lie derivative approx.
                # s = torch.sigmoid(a*(t_bias - t))
                # Lfb = a * s * (1 - s)             # time-varying Lie derivative approx.
                # t_bias = 5  # diffuser
                # s = torch.sigmod(t_bias-t)  # scalar
                # Lfb = -s * (1-s)

                if obs['type'] == 'circle':
                    b = dy**2 + dx**2 - s - self.robust_term
                    L1 = 2 * dy / self.yr
                    L2 = 2 * dx / self.xr
                elif obs['type'] == '4th':
                    b = dy**4 + dx**4 - s - self.robust_term
                    L1 = 4 * dy**3 / self.yr
                    L2 = 4 * dx**3 / self.xr
                else:
                    continue

                G_i = torch.cat([-L1, -L2], dim=1).unsqueeze(1)
                h_i = Lfb + b
            else:
                if obs['type'] == 'circle':
                    b      = dy**2 + dx**2 - 1 - self.robust_term
                    L1     = 2*dy**1 / self.yr
                    L2     = 2*dx**1 / self.xr
                elif obs['type'] == '4th':
                    b      = dy**4 + dx**4 - 1 - self.robust_term
                    L1     = 4*dy**3 / self.yr
                    L2     = 4*dx**3 / self.xr
                else:
                    continue

                if self.cbf_method == 'relax':
                    # add relax term: [rx1, rx0] or [0, sign]
                    sign = 100.0 if (t is not None and t <= self.relax_threshold) else 0.0
                    rx0 = torch.zeros_like(L1)
                    rx1 = sign * torch.ones_like(L1)
                    G_i = torch.cat([-L1, -L2, rx1, rx0], dim=1).unsqueeze(1)
                    h_i = b
                else:
                    # G_i: [B,1,2], h_i: [B,1]
                    G_i = torch.cat([-L1, -L2], dim=1).unsqueeze(1)
                    h_i = b

            G_list.append(G_i)
            h_list.append(h_i)
            safe_vals.append(torch.min(b + self.robust_term))

        # if you have no obstacles, just apply the reference control
        if not G_list:
            out = ref[:,2:4]
        else:
            G = torch.cat(G_list, dim=1)  # [B, num_obs, dim]
            h = torch.cat(h_list, dim=1)  # [B, num_obs]

            if self.cbf_solver == 'qp':
                if self.cbf_method == 'normal':
                    out = self.solve_qp(ref, G, h)
                elif self.cbf_method == 'relax':
                    out = self.solve_qp_relax(ref, G, h)
                elif self.cbf_method == 'time':
                    out = self.solve_qp(ref, G, h)
                else:
                    raise ValueError(f"Unknown CBF method '{self.cbf_method}'")
            # elif self.method == 'closed_form':
            #     TODO: out = self.solve_closed_form(ref, G, h)
            else:
                raise ValueError(f"Unknown CBF solver {self.cbf_solver}")
            
        # rebuild the next‐state
        rt = xp1.clone()
        rt[:,2:4] = x[:,2:4] + out[:, :2]
        return rt.unsqueeze(0), safe_vals
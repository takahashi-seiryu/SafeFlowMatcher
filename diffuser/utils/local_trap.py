import torch
import numpy as np

def local_trap(traj_tensor, cbf, batch_idx=0, n_timesteps=256):
    """
    Visualizes the distance between consecutive states in a trajectory and the CBF values.
    
    Args:
        traj_tensor: Tensor of shape [T, B, H, D] generated from p_sample_loop_ode_planning
        cbf: CBF object (includes obstacles and normalization information)
        batch_idx: Batch index to use
        n_timesteps: Horizon index to use (default is 256)
    """
    # Extract trajectory for a specific batch and horizon
    traj = traj_tensor[batch_idx, n_timesteps, :, 2:4]

    # Calculate distances between consecutive states
    distances = [0]
    for i in range(1, traj.shape[0]):
        dist = torch.norm(traj[i] - traj[i-1], p=2).item()
        distances.append(dist)

    # Calculate CBF values for each obstacle
    all_cbf_values = []
    for obs in cbf.obstacles:
        center = obs['center']
        n = obs['order']
        
        # Calculate CBF values
        cbf_values = []
        for i in range(traj.shape[0]):
            pos_y = traj[i, 0]
            pos_x = traj[i, 1]
            
            off_y = 2 * (center[1] - 0.5 - cbf.norm_mins[0]) / (cbf.norm_maxs[0] - cbf.norm_mins[0]) - 1
            off_x = 2 * (center[0] - 0.5 - cbf.norm_mins[1]) / (cbf.norm_maxs[1] - cbf.norm_mins[1]) - 1
            
            cbf_value = ((pos_y - off_y) / cbf.yr) ** n + ((pos_x - off_x) / cbf.xr) ** n - 1 - 0.01
            cbf_values.append(cbf_value.item())
        
        all_cbf_values.append(cbf_values)

    # Calculate the minimum CBF value for each obstacle
    cbf_values_array = np.array(all_cbf_values)  # Shape: [number of obstacles, number of time steps]
    min_cbf_values = np.min(cbf_values_array, axis=0)  # Shape: [number of time steps]

    DIST_THRESHOLD = 0.20
    CBF_THRESHOLD = 0.01
    num_trap = 0
    for i in range(1, traj.shape[0]):
        # if distance[i] > DIST_THRESHOLD and min_cbf_values[i] < CBF_THRESHOLD:
        if distances[i] > DIST_THRESHOLD:
            print(f"trapped at [{i}]| distance: {distances[i]}, cbf: {min_cbf_values[i]}")
            num_trap += 1
    
    return num_trap
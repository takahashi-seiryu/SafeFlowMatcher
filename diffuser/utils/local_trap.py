import torch
import numpy as np

def local_trap(traj_tensor, cbf, batch_idx=0, n_timesteps=256):
    """
    트래젝토리의 연속 상태 간 거리와 CBF 값을 시각화합니다.
    
    Args:
        traj_tensor: p_sample_loop_ode_planning에서 생성된 형태 [T, B, H, D]의 텐서
        cbf: CBF 객체 (obstacles와 normalization 정보 포함)
        batch_idx: 사용할 배치 인덱스
        n_timesteps: 사용할 호라이즌 인덱스 (기본값 256)
    """
    n_timesteps=256  #<------ fix this
    # 특정 배치와 호라이즌에 대한 트래젝토리 추출
    traj = traj_tensor[batch_idx, n_timesteps, :, 2:4]
    # 연속된 상태 간 거리 계산
    distances = []
    for i in range(0, traj.shape[0]-1):
        dist = torch.norm(traj[i] - traj[i+1], p=2).item()
        distances.append(dist)
    # extend to include the last distance    
    distances_long = []
    for i in range(0, traj.shape[0]):
        if i == 0 or i == traj.shape[0]-1:
            dist = 0
        else:
            dist = min(distances[i-1], distances[i])
        distances_long.append(dist)
    
    # 각 장애물에 대해 CBF 값 계산
    all_cbf_values = []
    for obs in cbf.obstacles:
        # 장애물 중심과 차수 추출
        center = obs['center']
        n = obs['order']
        
        # CBF 값 계산
        cbf_values = []
        for i in range(traj.shape[0]):
            # 위치 추출
            pos_y = traj[i, 0]
            pos_x = traj[i, 1]
            
            # 장애물 중심에서의 오프셋 계산
            off_y = 2 * (center[1] - 0.5 - cbf.norm_mins[0]) / (cbf.norm_maxs[0] - cbf.norm_mins[0]) - 1
            off_x = 2 * (center[0] - 0.5 - cbf.norm_mins[1]) / (cbf.norm_maxs[1] - cbf.norm_mins[1]) - 1
            
            # CBF 값 계산 (동일한 수식 사용)
            cbf_value = ((pos_y - off_y) / cbf.yr) ** n + ((pos_x - off_x) / cbf.xr) ** n - 1 - 0.01
            cbf_values.append(cbf_value.item())
        
        all_cbf_values.append(cbf_values)

    
    # 각 장애물에 대한 CBF 값의 최소값 계산
    cbf_values_array = np.array(all_cbf_values)  # 형태: [장애물 수, 시간 단계 수]
    min_cbf_values = np.min(cbf_values_array, axis=0)  # 형태: [시간 단계 수]

    dist_thr = 0.1
    cbf_thr = 0.05
    num_of_trap = 0
    for i in range(traj.shape[0]):
        #if num_of_trap > 1:
        #    break
        if distances_long[i] > dist_thr and min_cbf_values[i] < cbf_thr:
            #print(f"trapped: {i}/ distance: {distances_long[i]}, cbf: {min_cbf_values[i]}")
            num_of_trap += 1
    print(f"num of trap: {num_of_trap}")
    if num_of_trap == 0:
        trap1 = False
        trap2 = False
    elif num_of_trap == 1:
        trap1 = True
        trap2 = False
    else:
        trap1 = True
        trap2 = True
    
    return trap1, trap2
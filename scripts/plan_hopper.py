import json
import numpy as np
import os
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'      
    method: str = 'cfm'

args = Parser().parse_args('plan')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#
env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# CBF : 관절 각도 제한
args.obstacles = []  # 장애물 없음
args.cbf_solver = 'qp'
args.cbf_method = 'hard'
args.robust_term = 0.1
args.relax_threshold = 0.0


policy = Policy(diffusion, dataset.normalizer, args)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#---------------------------------- main loop ----------------------------------#
score_batch = []
comp_time = []
elbo_batch = []
success = 0

for iter in range(10):  
    print(f"step: {iter}/10")
    observation = env.reset()
    
    # Hopper의 목표: 앞으로 멀리 이동하기
    target_state = np.zeros_like(observation)
    target_state[0] = 5.0  # x-position 목표
    
    # 시작과 목표 상태 조건 설정
    cond = {
        0: observation,  # 시작 상태
        diffusion.horizon - 1: target_state  # 목표 상태
    }
    
    rollout = [observation.copy()]
    total_reward = 0
    
    # 경로 계획 생성
    start = time.time()
    action, samples, diffusion_paths, elbo = policy(cond, batch_size=args.batch_size)
    end = time.time()
    comp_time.append(end-start)
    elbo_batch.append(elbo)
    
    # 계획 시각화
    fullpath = os.path.join(args.savepath, f'{iter}.png')
    renderer.composite(fullpath, samples.observations, ncol=1)
    
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()
        
        if t < len(samples.actions[0]):
            action = samples.actions[0, t]
        else:
            action = samples.actions[0, -1]
        
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        
        print(f't: {t} | r: {reward:.2f} | total: {total_reward:.2f}')
        
        rollout.append(next_observation.copy())
        
        # 매 10 스텝마다 롤아웃 시각화
        if t % 10 == 0 or terminal:
            renderer.composite(os.path.join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
        
        if terminal:
            break

        observation = next_observation
    
    # Hopper 성공 기준: 높은 보상
    if total_reward > 1000: 
        success += 1
    
    score_batch.append(total_reward)

print("score mean:", np.mean(score_batch))
print("score std:", np.std(score_batch))
print("computation time:", np.mean(comp_time))
print("success rate:", success/10)

json_path = os.path.join(args.savepath, 'results.json')
json_data = {
    'score': float(np.mean(score_batch)),
    'computation_time': float(np.mean(comp_time)),
    'success_rate': float(success/10),
    'epoch_diffusion': diffusion_experiment.epoch
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
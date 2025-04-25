import json
import numpy as np
import os
import time
import imageio
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'      
    method: str = 'cfm'
    logbase: str = 'logs'
    diffusion_loadpath: str = None  
    diffusion_epoch: str = 'latest'

args = Parser().parse_args('plan')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args.diffusion_loadpath = os.path.join('hopper', args.method)  # 'hopper/cfm'

#---------------------------------- setup ----------------------------------#
env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#
# Ensure the model path exists
model_path = os.path.join(args.logbase, args.diffusion_loadpath, args.dataset)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path not found: {model_path}. Please ensure you have trained the model first.")

diffusion_experiment = utils.load_diffusion(
    args.logbase,
    args.diffusion_loadpath,  # 'hopper/cfm'
    args.dataset,             # 'hopper-medium-v2'
    epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

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
    
    cond = {
        0: observation,  
        diffusion.horizon - 1: target_state  # 목표 상태
    }
    
    rollout = [observation.copy()]
    total_reward = 0
    frames = []
    
    start = time.time()
    action, samples, diffusion_paths, elbo = policy(cond, batch_size=args.batch_size)
    end = time.time()
    comp_time.append(end-start)
    elbo_batch.append(elbo)
    
    fullpath = os.path.join(args.savepath, f'{iter}.png')
    # renderer.composite(fullpath, samples.observations, ncol=1)
    renderer.composite(fullpath, samples.observations)  

    for t in range(diffusion.max_episode_steps):
        state = env.state_vector().copy()
        
        if t < len(samples.actions[0]):
            action = samples.actions[0, t]
        else:
            action = samples.actions[0, -1]
        
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        
        normalized_reward = reward / 1000.0  # Hopper environment의 reward scale 조정
        normalized_total = total_reward / 1000.0
        
        print(f't: {t} | r: {reward:.2f} (norm: {normalized_reward:.3f}) | total: {normalized_total:.3f} | raw: {total_reward:.2f}')
        
        state = env.state_vector()
        img = renderer.render(state)
        frames.append(img)
        
        rollout.append(next_observation.copy())
        
        if t % 10 == 0 or terminal:
            renderer.composite(os.path.join(args.savepath, 'rollout.png'), np.array(rollout)[None])
        
        if terminal:
            print("terminated")
            break

        observation = next_observation
    
    if total_reward > 1000: 
        success += 1
    
    normalized_score = total_reward / 1000.0  # Normalize score for paper comparison
    score_batch.append(normalized_score)
    print(f'Episode {iter} | Score: {normalized_score:.3f} | Raw Score: {total_reward:.2f}')
    
    mp4_path = os.path.join(args.savepath, f'rollout_{iter}.mp4')
    print(f'Saving rollout to {mp4_path}')
    print(f"[DEBUG] Total frames to save: {len(frames)}")
    imageio.mimsave(mp4_path, frames, fps=20)

print('='*50)
print(f'Mean Score: {np.mean(score_batch):.3f} ± {np.std(score_batch):.3f}')  # Normalized scores
print(f'Mean Computation Time: {np.mean(comp_time):.4f} ± {np.std(comp_time):.4f}')
print(f'Mean ELBO: {np.mean(elbo_batch):.4f} ± {np.std(elbo_batch):.4f}')
print(f'Success Rate: {success/10:.2f}')

json_path = os.path.join(args.savepath, 'results.json')
json_data = {
    'score': float(np.mean(score_batch)),
    'computation_time': float(np.mean(comp_time)),
    'success_rate': float(success/10),
    'epoch_diffusion': diffusion_experiment.epoch
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
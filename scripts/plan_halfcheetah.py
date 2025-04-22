import json
import numpy as np
from os.path import join
import os
import time

import diffuser.utils as utils
import torch

# python scripts/plan_halfcheetah.py --model_loadpath /home/work/CBF-CFM/ziwon/CBF-CFM/logs/halfcheetah --model_epoch latest

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'
    # Paths
    savepath: str = 'logs/halfcheetah/eval'
    
    # Model loading
    model_loadpath: str = None  # Path to the trained model directory
    model_epoch: str = 'latest'  # Which epoch to load ('latest' or specific number)
    batch_size: int = 1  # Batch size for planning
    device: str = 'cuda'  # Device to run inference on

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('diffusion')

#---------------------------------- loading ----------------------------------#

# Load the trained model
if args.model_loadpath is None:
    raise ValueError("Please provide the path to the trained model using --model_loadpath")

# Load model configs and weights
model_path = args.model_loadpath
if args.model_epoch == 'latest':
    # Find the latest epoch
    model_files = [f for f in os.listdir(model_path) if f.startswith('state_') and f.endswith('.pt')]
    if not model_files:
        raise ValueError(f"No model checkpoints found in {model_path}")
    latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in model_files])
    model_file = f'state_{latest_epoch}.pt'
else:
    model_file = f'state_{args.model_epoch}.pt'

# Load the full checkpoint
checkpoint = torch.load(os.path.join(model_path, model_file), map_location=args.device)

# Load dataset and configs
dataset_config = utils.Config.load(os.path.join(model_path, 'dataset_config.pkl'))
render_config = utils.Config.load(os.path.join(model_path, 'render_config.pkl'))
diffusion_config = utils.Config.load(os.path.join(model_path, 'diffusion_config.pkl'))
model_config = utils.Config.load(os.path.join(model_path, 'model_config.pkl'))

# Create dataset and renderer
dataset = dataset_config()
renderer = render_config()

# Create model and load state
model = model_config()
diffusion = diffusion_config(model)
diffusion.load_state_dict(checkpoint['model'])
diffusion.to(args.device)
diffusion.eval()

# Create policy
policy = utils.Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#---------------------------------- main loop ----------------------------------#
score_batch = []
comp_time = []
success = 0

# Create save directories
makedirs(args.savepath)
makedirs(join(args.savepath, 'png'))

for iter in range(10):  # Run 10 evaluation episodes
    print(f"Episode: {iter+1}/10")

    # Reset environment
    env = dataset.env
    observation = env.reset()

    # Initialize rollout for storing trajectory
    rollout = [observation.copy()]
    
    # Track rewards
    total_reward = 0
    
    # Plan once at the beginning (open-loop control)
    start = time.time()
    
    # Set initial condition
    cond = {0: observation}
    
    # Generate plan
    action, samples, diffusion_paths = policy(cond, batch_size=args.batch_size)
    
    end = time.time()
    comp_time.append(end-start)
    
    # Get planned actions and observations
    actions = samples.actions[0]
    sequence = samples.observations[0]
    
    # Save trajectory visualization
    fullpath = join(args.savepath, f'trajectory_{iter}.png')
    renderer.composite(fullpath, samples.observations, ncol=1)
    
    # Save diffusion process as video
    renderer.render_diffusion(join(args.savepath, f'diffusion_{iter}.mp4'), diffusion_paths[0])
    
    # Save individual frames of diffusion process
    diff_step = diffusion_paths[0].shape[0]
    for kk in range(diff_step):
        imgpath = join(args.savepath, f'png/diffusion_{iter}_step_{kk}.png')
        renderer.composite(imgpath, diffusion_paths[0][kk:kk+1], ncol=1)
    
    # Execute the plan in the environment
    for t in range(min(env._max_episode_steps, sequence.shape[0]-1)):
        # Get next action from planned trajectory
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
        
        # Use simple controller to track the planned trajectory
        state = env.state_vector().copy()
        action = next_waypoint[:dataset.action_dim] 
        
        # Take step in environment
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        
        # Calculate normalized score
        score = env.get_normalized_score(total_reward)
        
        print(f't: {t} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f}')
        
        # Save to rollout for visualization
        rollout.append(next_observation.copy())
        
        # Render rollout at regular intervals
        if t % 20 == 0 or terminal:
            # Save current rollout
            rollout_path = join(args.savepath, f'rollout_{iter}.png')
            renderer.composite(rollout_path, np.array(rollout)[None], ncol=1)
            
            # Save as video
            rollout_video_path = join(args.savepath, f'rollout_{iter}.mp4')
            renderer.render_rollout(rollout_video_path, rollout, fps=30)
        
        if terminal:
            break
            
        observation = next_observation
    
    # Track success rate (assuming success is reaching a score above 0.6)
    if score > 0.6:
        success += 1
    
    score_batch.append(score)
    
    # Save final rollout
    rollout_path = join(args.savepath, f'rollout_{iter}.png')
    renderer.composite(rollout_path, np.array(rollout)[None], ncol=1)
    
    rollout_video_path = join(args.savepath, f'rollout_{iter}.mp4')
    renderer.render_rollout(rollout_video_path, rollout, fps=30)

# Print statistics
score_batch = np.array(score_batch)
comp_time = np.array(comp_time)

print("\n----- Evaluation Results -----")
print(f"Score mean: {np.mean(score_batch):.4f}")
print(f"Score std: {np.std(score_batch):.4f}")
print(f"Planning time: {np.mean(comp_time):.4f} seconds")
print(f"Success rate: {success}/10")

# Save results as a JSON file
json_path = join(args.savepath, 'results.json')
json_data = {
    'scores': score_batch.tolist(),
    'score_mean': float(np.mean(score_batch)),
    'score_std': float(np.std(score_batch)),
    'computation_time': float(np.mean(comp_time)),
    'success_rate': int(success),
    'epoch_diffusion': args.model_epoch
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

print(f"\nResults saved to {args.savepath}")
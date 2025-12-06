import pdb
import os
import torch
import diffuser.sampling as sampling
import diffuser.utils as utils
# from gym.wrappers import RecordEpisodeStatistics, RecordVideo  # removed
import imageio
from diffuser.models.temporal import ValueFunction
from diffuser.sampling.guides import ValueGuide
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # dataset: str = 'walker2d-medium-replay-v2'
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'
    loadbase: str = 'logs'
    preprocess_fns: list = []
    
    # add cov-g related arguments
    guidance_type: str = 'direct'  # choose among 'direct', 'use_learned_v', 'rw'
    guidance_scale: float = 1.0    # adjust guidance strength
    value_model_path: str = None   # value function model path (create new if missing)
    value_hidden_dim: int = 256    # hidden dimension for the value model

args = Parser().parse_args('plan')

args.batch_size = 1
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create and initialize value function model
value_model = ValueFunction(
    horizon=diffusion.horizon,
    transition_dim=diffusion.transition_dim,
    cond_dim=args.value_hidden_dim
).to(device)

# load value model if a path is provided
if args.value_model_path and os.path.exists(args.value_model_path):
    value_model.load_state_dict(torch.load(args.value_model_path))
    print(f"Loaded value model from {args.value_model_path}")
else:
    print("Using untrained value model")

# create value-function guide
value_guide = ValueGuide(model=value_model).to(device)

# enable reward guidance using the cov-g method
diffusion.enable_guidance(
    value_model=value_model,
    guidance_type=args.guidance_type,
    scale=args.guidance_scale
)

# create log directory
os.makedirs(args.savepath, exist_ok=True)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)  

# use the updated GuidedPolicy class
from diffuser.sampling.policies import GuidedPolicy

policy_config = utils.Config(
    GuidedPolicy,
    guide=value_guide,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    args=args
)

logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
# RecordVideo wrapper removed; use simple imageio instead

# create videos directory
os.makedirs("videos", exist_ok=True)

comp_time = []
safety = []
scores = []
import time

for kk in range(5):
    observation = env.reset()
    frames = [env.render(mode='rgb_array')] 
    ## observations for rendering
    rollout = [observation.copy()]
    total_reward = 0
    
    for t in range(args.max_episode_length): #max = 1000
    # for t in range(4): #max = 1000
        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}

        start = time.time()
        action, trajectories, diffusion_paths, b_min = policy(conditions, batch_size=args.batch_size, verbose=(t==0))
        end = time.time()
        
        # use sum_elbo (or another metric) instead of the safety value (b_min)
        if t == 0:
            # safety.append(b_min.cpu())
            safety.append(b_min)
            comp_time.append(end-start)
            
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        frames.append(env.render(mode='rgb_array'))  

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f'step: {kk}/1 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'guidance_type: {args.guidance_type} | scale: {args.guidance_scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())
        if t==0:
            logger.log(t, trajectories, state, rollout, diffusion_paths)

        if terminal:
            logger.log(t, trajectories, state, rollout, diffusion_paths)
            break

        observation = next_observation
    scores.append(score)
    
    # save video
    # video_path = f'videos/{args.dataset}_rollout_{kk}.mp4'
    # imageio.mimsave(video_path, frames, fps=30)
    # print(f"Video saved to: {video_path}")

## write results to json file at `args.savepath`
import numpy as np
comp_time = np.array(comp_time)
# safety = np.array(safety)
scores = np.array(scores)
# print("safety: ", np.min(safety))
print("score mean: ", np.mean(scores))
print("score std: ", np.std(scores))
print("computation time: ", np.mean(comp_time))
print(f"Guidance type: {args.guidance_type}, Guidance scale: {args.guidance_scale}")

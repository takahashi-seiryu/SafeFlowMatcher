import diffuser.utils as utils
import os
import pdb
import time
import numpy as np
from collections import namedtuple
import torch
import einops

Trajectories = namedtuple('Trajectories', 'actions observations values')

class Policy:
    def __init__(self, diffusion_model, normalizer, preprocess_fns, scale=1.0, verbose=False, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        
        if callable(preprocess_fns):
            self.preprocess_fn = preprocess_fns
        else:
            self.preprocess_fn = preprocess_fns[0] if isinstance(preprocess_fns, list) and len(preprocess_fns) > 0 else lambda x: x
        
        self.sample_kwargs = sample_kwargs
        self.scale = scale
        self.verbose = verbose

    def __call__(self, conditions, batch_size=1, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        if hasattr(self.normalizer.normalizers['observations'], 'mins'):
            mins = self.normalizer.normalizers['observations'].mins
            maxs = self.normalizer.normalizers['observations'].maxs
            
            if hasattr(self.diffusion_model, 'mean'):
                self.diffusion_model.mean = (mins + maxs) / 2
                self.diffusion_model.std = (maxs - mins) / 2
        
        try:
            shape = (batch_size, self.diffusion_model.horizon, self.diffusion_model.transition_dim)
            
            samples = self.diffusion_model.conditional_sample(
                conditions,
                record_traj=True,
                verbose=verbose,
                **self.sample_kwargs
            )
            
            if isinstance(samples, tuple) and len(samples) == 2:
                samples, diffusion_chains = samples
            else:
                diffusion_chains = None
                
        except TypeError as e:
            if "record_traj" in str(e):
                print("Warning: record_traj not supported, trying without it")
                samples = self.diffusion_model.conditional_sample(
                    conditions,
                    verbose=verbose,
                    **self.sample_kwargs
                )
                diffusion_chains = None
            else:
                print(f"Error during conditional_sample: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error during conditional_sample: {e}")
            raise

        b_min = getattr(self.diffusion_model, 'safe1', None)
        if b_min is None:
            b_min = 0.0
        
        try:
            trajectories = utils.to_np(samples)
            
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')
            action = actions[0, 0]

            normed_observations = trajectories[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')

            if diffusion_chains is not None:
                try:
                    diffusion_obs = utils.to_np(diffusion_chains)
                    if len(diffusion_obs.shape) >= 4:
                        diffusion_obs = diffusion_obs[:, :, :, self.action_dim:]
                        diffusion_obs = self.normalizer.unnormalize(diffusion_obs, 'observations')
                    else:
                        diffusion_obs = None
                except:
                    diffusion_obs = None
            else:
                diffusion_obs = None

            values = None
            trajectories = Trajectories(actions, observations, values)
            
            return action, trajectories, diffusion_obs, b_min
            
        except Exception as e:
            print(f"Error processing samples: {e}")
            dummy_action = np.zeros(self.action_dim)
            dummy_obs = np.zeros((batch_size, self.diffusion_model.horizon, self.diffusion_model.observation_dim))
            dummy_trajectories = Trajectories(dummy_action, dummy_obs, None)
            return dummy_action, dummy_trajectories, None, 0.0

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _format_conditions(self, conditions, batch_size):
        try:
            normalized_conditions = {}
            for k, v in conditions.items():
                if isinstance(v, np.ndarray):
                    v_tensor = torch.from_numpy(v).float()
                elif isinstance(v, torch.Tensor):
                    v_tensor = v.float()
                else:
                    v_tensor = torch.tensor(v, dtype=torch.float)
                    
                normalized_v = self.normalizer.normalize(v_tensor, 'observations')
                normalized_conditions[k] = normalized_v
                
            device_conditions = {}
            for k, v in normalized_conditions.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=torch.float32)
                v = v.to(self.device)
                
                if batch_size > 1:
                    v = v.unsqueeze(0).repeat(batch_size, *[1] * len(v.shape))
                device_conditions[k] = v
                
            return device_conditions
        except Exception as e:
            print(f"Error in _format_conditions: {e}")
            return conditions

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'
    loadbase: str = 'logs'

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

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

logger = logger_config()

#-----------------------------------------------------------------------------#
#--------------------------- planning: diffusion-based ----------------------#
#-----------------------------------------------------------------------------#

policy = Policy(
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    scale=args.scale,
    verbose=args.verbose
)

env = dataset.env
comp_time = []
safety = []
scores = []


import os
render_dir = os.path.join(args.savepath, 'renders')
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

print("Starting execution loop...")
for kk in range(10):
    try:
        observation = env.reset()
        print(f"Reset environment for episode {kk}")

        rollout = [observation.copy()]
        total_reward = 0
        score = 0
        
        episode_frames = []
        
        for t in range(args.max_episode_length):
            
            state = env.state_vector().copy()
            conditions = {0: observation}
            
            start = time.time()
            try:
                action, samples, diffusion, b_min = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
                end = time.time()
                
                if t == 0:
                    if b_min is not None:
                        if isinstance(b_min, torch.Tensor):
                            safety.append(b_min.cpu().numpy())
                        elif isinstance(b_min, np.ndarray):
                            safety.append(b_min)
                        else:
                            safety.append(np.array([float(b_min)]))
                    else:
                        safety.append(np.array([0.0]))
                    comp_time.append(end-start)

                next_observation, reward, terminal, _ = env.step(action)

                total_reward += reward
                score = env.get_normalized_score(total_reward)
                
                values_display = "N/A"
                if hasattr(samples, 'values') and samples.values is not None:
                    values_display = samples.values
                
                print(f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | values: {values_display} | scale: {args.scale}', flush=True)

                rollout.append(next_observation.copy())
                
                logger.log(t, samples, state, rollout, diffusion)

                if hasattr(renderer, 'render_rollout'):
                    renderer.render_rollout(
                        os.path.join(self.savepath, 'rollout.mp4'),
                        rollout
                    )

                if terminal:
                    print(f"Episode {kk} terminated at step {t}")
                    break

                observation = next_observation
            except Exception as e:
                print(f"Error during execution: {e}")
                import traceback
                traceback.print_exc()
                break
                
        if episode_frames and hasattr(renderer, 'save_video'):
            print(f"에피소드 {kk} 렌더링 저장 중...")
            video_path = os.path.join(render_dir, f'episode_{kk}.mp4')
            renderer.save_video(episode_frames, video_path)
            print(f"렌더링 저장됨: {video_path}")
            
        scores.append(score)
        print(f"Episode {kk} completed - total reward: {total_reward:.2f}, normalized score: {score:.4f}")
        
    except Exception as e:
        print(f"Error in episode {kk}: {e}")
        import traceback
        traceback.print_exc()
        continue

results_file = os.path.join(args.savepath, 'planning_results.json')

results = {
    'env': args.dataset,
    'method': args.method,
    'scale': args.scale,
    'scores': [float(s) for s in scores],
    'safety': float(np.min(safety)) if len(safety) > 0 and safety.size > 0 else "N/A",
    'score_mean': float(np.mean(scores)) if len(scores) > 0 else "N/A",
    'score_std': float(np.std(scores)) if len(scores) > 0 else "N/A",
    'computation_time': float(np.mean(comp_time)) if len(comp_time) > 0 else "N/A"
}

print("\n========== Final Results ==========")
print("safety: ", np.min(safety) if len(safety) > 0 and safety.size > 0 else "N/A")
print("score mean: ", np.mean(scores) if len(scores) > 0 else "N/A")
print("score std: ", np.std(scores) if len(scores) > 0 else "N/A")
print("computation time: ", np.mean(comp_time) if len(comp_time) > 0 else "N/A")

import json
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nResults saved to {results_file}")

logger.finish(
    args.max_episode_length, 
    np.mean(scores) if len(scores) > 0 else 0, 
    np.mean([s for s in total_reward if s is not None]) if 'total_reward' in locals() else 0, 
    terminal,
    diffusion_experiment, 
    None 
)
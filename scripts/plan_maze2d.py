import json
import numpy as np
from os.path import join, dirname
import pdb
import csv
import time
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

# fix seed
import random
seed = 42#42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# fix seed

# diffusion model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method base
# cfm model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm
# cfm model with safety disabled:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm --safety false
# cfm model with safety enabled:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm --safety true

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    method: str = 'cfm'
    safety: str = None  # Override safety_enabled: 'true', 'false', or None (use config)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# Override safety_enabled if --safety flag is provided
if args.safety is not None:
    args.safety_enabled = args.safety.lower() == 'true'
    print(f"[Override] safety_enabled = {args.safety_enabled}")

utils.set_device(args.device)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch, device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer, args)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy

# --------------------------------- csv header ----------------------------------#
# save results to csv file
csv_path = join(args.savepath, 'results.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['iter', 'safe1', 'safe2', 'trap1', 'trap2', 'score', 'itertime', 'success'])

#---------------------------------- helper functions ----------------------------------#
def check_position_safety(pos, obstacles, norm_mins, norm_maxs):
    """
    Check if a position is safe (outside all obstacles).
    Returns True if safe, False if inside any obstacle.
    pos: (y, x) position in original coordinates
    """
    yr = 2 / (norm_maxs[0] - norm_mins[0])
    xr = 2 / (norm_maxs[1] - norm_mins[1])

    for obs in obstacles:
        cx, cy = obs['center']
        order = obs['order']

        off_x = 2 * (cx - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1
        off_y = 2 * (cy - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1

        # Normalize position
        pos_y_norm = 2 * (pos[0] - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1
        pos_x_norm = 2 * (pos[1] - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1

        dx = (pos_x_norm - off_x) / xr
        dy = (pos_y_norm - off_y) / yr

        # CBF value: positive means safe (outside obstacle)
        cbf_value = dy**order + dx**order - 1

        if cbf_value < 0:
            return False  # Inside this obstacle

    return True  # Outside all obstacles

#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
score_batch = []
elbo_batch = []
is_trap1, is_trap2 = 0, 0
num_trap1, num_trap2 = 0, 0
c_smooth_batch, s_smooth_batch = [], []
num_success = 0
num_safe_total = 0  # Counter for trials where both safe1 and safe2 are positive
iter_time_batch = []
plan_total_time_batch = []  # Track planning time only (not rollout)
is_save_vid = False  # if you want to save diffusion process video
is_save_img = True   # if you want to save result trajectories
is_save_rollout = False  # if you want to save rollout image
is_save_img_group = False  # if you want to save diffusion process images
set_cond = False

# Get normalization parameters for safety check
norm_mins = dataset.normalizer.normalizers['observations'].mins
norm_maxs = dataset.normalizer.normalizers['observations'].maxs

TOTAL_TEST_ITER = 100
for iter in range(1, TOTAL_TEST_ITER+1):   # num of testing runs
    print("Total test iteration: ", iter, f"/{TOTAL_TEST_ITER}")

    # Random start-goal with safety check
    max_resample_attempts = 100
    resample_count = 0

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    while True:
        if set_cond:
            # Get random start position
            observation = env.reset()
            # Set random target (goal)
            env.set_target()
            target = env._target

        # If safety is enabled, check if start and goal are safe
        if set_cond and args.safety_enabled and hasattr(args, 'obstacles') and args.obstacles:
            start_pos = observation[:2]  # (y, x)
            goal_pos = target  # (y, x)

            start_safe = check_position_safety(start_pos, args.obstacles, norm_mins, norm_maxs)
            goal_safe = check_position_safety(goal_pos, args.obstacles, norm_mins, norm_maxs)

            if start_safe and goal_safe:
                break  # Both positions are safe
            else:
                resample_count += 1
                if resample_count >= max_resample_attempts:
                    print(f"Warning: Could not find safe start-goal pair after {max_resample_attempts} attempts. Using current positions.")
                    break
        else:
            break  # No safety check needed

    if resample_count > 0:
        print(f"  Resampled {resample_count} times to find safe start-goal pair")

    # print(f"  Start: {observation[:2]}, Goal: {target}")

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    env_step = env.max_episode_steps
    # for t in range(env_step):
    for t in range(800): #real is 384

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            # Measure planning time only
            plan_start_time = time.time()
            action, samples, diffusion_paths, plan_safe1, plan_safe2, elbo, num_trap, iter_time, c_smooth, s_smooth = policy(cond, batch_size=args.batch_size)
            plan_total_time = time.time() - plan_start_time

            elbo_batch.append(elbo)
            # Note: plan_safe1, plan_safe2 are based on planned trajectory (not used for statistics)
            
            actions = samples.actions[0]
            sequence = samples.observations[0]
            velocities = (sequence[1:, :2] - sequence[:-1, :2]) # derive velocity from consecutive positions
            diffusion_paths = diffusion_paths[0]
            disp_mag = np.linalg.norm(velocities, axis=1)   # shape: (383,)
            ##################################################start saving videos/images
            # Save the composite image of all trajectories
            if is_save_img: # if you want to save result trajectories
                fullpath = join(args.savepath, f'results/{iter}.png')
                os.makedirs(dirname(fullpath), exist_ok=True)
                renderer.composite(fullpath, samples.observations, ncol=1)
                
                # Save the next step composite image of all trajectoiries
                """
                for checking velocity of each way-points, 
                """
                # fullpath = join(args.savepath, f'{iter}_next.png')
                # next_time_observation = samples.observations.copy()
                # for i in range(next_time_observation.shape[1]-1):
                #     next_time_observation[:, i, :2] += samples.observations[:, i+1, 2:4] * 1e-2   # next pos = curr pos + next vel * 1e-2(time interval)
                # renderer.composite(fullpath, next_time_observation, ncol=1)

            # Save the diffusion process as a video
            if is_save_vid: # if you want to save diffusion process video
                renderer.render_diffusion(join(args.savepath, f'diffusion_{iter}.mp4'), diffusion_paths)
            
            if False: # if you want to save if plan be played
                i = np.arange(384)[:, None]
                j = np.arange(384)[None, :]
                src = np.minimum(i, j)
                diffusion_paths_video = diffusion_paths[-1][src]
                renderer.render_diffusion(join(args.savepath, f'diffusion_if_play_{iter}.mp4'), diffusion_paths_video)
            
            
            # Save individual frames of the diffusion process
            if is_save_img_group: # if you want to save diffusion process images
                diff_step = diffusion_paths.shape[0]  
                makedirs(join(args.savepath, 'png'))
                for kk in range(diff_step):
                    imgpath = join(args.savepath, f'png/{kk}.png')
                    renderer.composite(imgpath, diffusion_paths[kk:kk+1], ncol=1)
            
            # Save individual state's movement of the diffusion process
            # diff_step = diffusion_paths.shape[0]  
            # makedirs(join(args.savepath, 'state'))
            # for k in range(258):
            #     imgpath = join(args.savepath, f'state/{k}.png')
            #     renderer.composite_state(imgpath, diffusion_paths[:,k], ncol=1)
            #################################################end saving videos/images

        ##################################################start planning
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        # can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        if t==0:
            prev_action = action.copy()
            prev_state = state.copy()
        # if t < 400:
        #     print(f"t: {t} | action: {action} | next_waypoint: {next_waypoint} | state: {state}")
        #     print(f"v dt: {(state[2] - prev_state[2]) / prev_action[0]} / {(state[3] - prev_state[3]) /prev_action[1]}")
        #     print(f"x dt: {(state[0] - prev_state[0]) / state[2]} / {(state[1] - prev_state[1]) / state[3]}")
        prev_action = action.copy()
        prev_state = state.copy()

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        ##################################################end planning

        ##################################################start logging
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        # if 'maze2d' in args.dataset:
        #     xy = next_observation[:2]
        #     goal = env.unwrapped._target
        #     print(
        #         f'maze | pos: {xy} | goal: {goal}'
        #     )

        ## update rollout observations
        rollout.append(next_observation.copy())
        #print(len(rollout), rollout[0].shape, [rollout[0]])

        # logger.log(score=score, step=t)
        ##################################################end logging

        ##################################################start saving videos/images
        # if t % args.vis_freq == 0 or terminal:
        #     fullpath = join(args.savepath, f'{t}.png')

        #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        #     # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        #     ## save rollout thus far
        #     renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)   ## debug

        #     # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        #     # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)
        ##################################################end saving videos/images

        if terminal:
            break

        observation = next_observation
    
    ######################
    arr = np.expand_dims(np.stack(rollout), axis=0)
    if is_save_rollout: # if you want to save rollout image
        fullpath = join(args.savepath, f'rollout_{iter}.png')
        renderer.composite(fullpath, arr, ncol=1)

    ##################################################start rollout-based safety calculation
    # Compute safety based on actual rollout trajectory (not planned trajectory)
    rollout_arr = np.stack(rollout)  # shape: [T, 4] where 4 = (y, x, vy, vx)
    # Normalize the rollout observations
    rollout_normalized = dataset.normalizer.normalize(rollout_arr, 'observations')
    # cbf_nv expects [batch, T, action_dim + obs_dim] format where action_dim=2
    # The sample format is [action1, action2, obs_y, obs_x, obs_vy, obs_vx]
    # Add dummy actions (zeros) to match the expected format
    dummy_actions = np.zeros((rollout_normalized.shape[0], 2))  # [T, 2]
    rollout_with_actions = np.concatenate([dummy_actions, rollout_normalized], axis=1)  # [T, 6]
    # Convert to tensor with batch dimension: [1, T, 6]
    rollout_tensor = torch.tensor(rollout_with_actions, dtype=torch.float32, device=args.device).unsqueeze(0)
    # Compute safety values using CBF (based on actual rollout)
    rollout_safe_l = policy.diffusion_model.cbf.cbf_nv(rollout_tensor)
    safe1_val, safe2_val = rollout_safe_l[0], rollout_safe_l[1]
    ##################################################end rollout-based safety calculation

    ##################################################start statistics calculation
    if reward > 0.95:
        num_success = num_success + 1

    if num_trap >= 1:
        is_trap1 = 1
        num_trap1 += 1
    else:
        is_trap1 = 0
    
    if num_trap >= 2:
        is_trap2 = 1
        num_trap2 += 1
    else:
        is_trap2 = 0

    safe1_batch.append(safe1_val)
    safe2_batch.append(safe2_val)
    score_batch.append(score)
    iter_time_batch.append(iter_time)
    c_smooth_batch.append(c_smooth.item())
    s_smooth_batch.append(s_smooth.item())

    # Count safe trials: safe only if both safe1 >= 0 AND safe2 >= 0
    if safe1_val >= 0 and safe2_val >= 0:
        num_safe_total += 1
    ##################################################end statistics calculation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ##################################################start per-iter statistics calculation
    # per-iter statistics calculation
    score_val    = float(score)
    itertime_val = float(iter_time[0])
    success_val  = 1 if reward > 0.95 else 0

    # for time saving skip it
    # append mode to write one line to csv file
    # with open(csv_path, 'a', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow([
    #         iter,
    #         safe1_val,
    #         safe2_val,
    #         is_trap1,
    #         is_trap2,
    #         score_val,
    #         itertime_val,
    #         success_val
    #     ])

    # Track planning time only
    plan_total_time_batch.append(plan_total_time)
    ##################################################end per-iter statistics calculation

# --------------------------------- statistics calculation ----------------------------------#

iter_time_batch = np.array(iter_time_batch)
iter_time_avg = np.mean(iter_time_batch, axis=0)

# Calculate average planning time
plan_total_time_batch = np.array(plan_total_time_batch)
avg_plan_time = np.mean(plan_total_time_batch)

# Calculate safe_total_rate
safe_total_rate = num_safe_total / TOTAL_TEST_ITER if TOTAL_TEST_ITER > 0 else 0.0

# pdb.set_trace()
print("======================results======================")
# elbo_batch = np.array(elbo_batch)
# print("elbo mean: ", np.mean(elbo_batch))
# print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
print(f"safe1: min: {np.min(safe1_batch):.4f}/ {np.mean(safe1_batch):.4f} ± {np.std(safe1_batch):.4f}")
print(f"safe2: min: {np.min(safe2_batch):.4f}/ {np.mean(safe2_batch):.4f} ± {np.std(safe2_batch):.4f}")
print(f"safe_total_rate: {num_safe_total} / {TOTAL_TEST_ITER} = {safe_total_rate:.4f}")
print(f"trap1: {num_trap1} / {TOTAL_TEST_ITER}")
print(f"trap2: {num_trap2} / {TOTAL_TEST_ITER}")
print(f"c-smooth: {np.mean(c_smooth_batch):.4f} ± {np.std(c_smooth_batch):.4f}")
print(f"s-smooth: {np.mean(s_smooth_batch):.4f} ± {np.std(s_smooth_batch):.4f}")
print(f"score: {np.mean(score_batch):.5f} ± {np.std(score_batch):.5f}")
print(f"number of success: {num_success} / {TOTAL_TEST_ITER}")
print("--------------------- timing ---------------------")
print(f"[Plan] Total time (avg): {avg_plan_time*1000:.2f} ms ± {np.std(plan_total_time_batch)*1000:.2f} ms")
print("=======================end=========================")

# --------------------------------- plot ----------------------------------#
try:
    import matplotlib.pyplot as plt
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot safe1 data
    ax1.set_title('Safety Specification 1')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Safety Value')
    ax1.plot(range(len(safe1_batch)), safe1_batch, 'b-', label='Safe1')
    ax1.grid(True)
    ax1.legend()
    
    # Plot safe2 data
    ax2.set_title('Safety Specification 2')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Safety Value')
    ax2.plot(range(len(safe2_batch)), safe2_batch, 'r-', label='Safe2')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    imgpath = join(args.savepath, 'safety_stats.png')
    plt.savefig(imgpath, dpi=300, bbox_inches='tight')
    plt.close('all')  # Properly close all figures
    
except Exception as e:
    print(f"Warning: Could not create plots: {str(e)}")

# Save result as a json file
try:
    json_path = join(args.savepath, 'rollout.json')
    json_data = {
        'safety_stats': {
            'safe1_mean': float(np.mean(safe1_batch)),
            'safe2_mean': float(np.mean(safe2_batch)),
            'safe1_std': float(np.std(safe1_batch)),
            'safe2_std': float(np.std(safe2_batch)),
            'safe1_min': float(np.min(safe1_batch)),
            'safe2_min': float(np.min(safe2_batch)),
            'safe_total_count': num_safe_total,
            'safe_total_rate': float(safe_total_rate)
        },
        'smooth_stats': {
            'c_smooth_mean': float(np.mean(c_smooth_batch)),
            's_smooth_mean': float(np.mean(s_smooth_batch)),
            'c_smooth_std': float(np.std(c_smooth_batch)),
            's_smooth_std': float(np.std(s_smooth_batch)),
        },
        'score_stats': {
            'mean': float(np.mean(score_batch)),
            'std': float(np.std(score_batch)),
            'min': float(np.min(score_batch)),
            'max': float(np.max(score_batch))
        },
        'planning_time_ms': {
            'mean': float(avg_plan_time * 1000),
            'std': float(np.std(plan_total_time_batch) * 1000),
            'min': float(np.min(plan_total_time_batch) * 1000),
            'max': float(np.max(plan_total_time_batch) * 1000)
        },
        'local_trap1': f'{num_trap1}/{TOTAL_TEST_ITER}',
        'local_trap2': f'{num_trap2}/{TOTAL_TEST_ITER}',
        'denoising_step': args.n_diffusion_steps,
        # 'step': int(t),
        # 'return': float(total_reward),
        # 'term': bool(terminal),
        # 'epoch_diffusion': int(diffusion_experiment.epoch),
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)
except Exception as e:
    print(f"Warning: Could not save json file: {str(e)}")

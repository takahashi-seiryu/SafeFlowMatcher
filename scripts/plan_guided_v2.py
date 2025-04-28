# import pdb
# import os
# import torch
# import diffuser.sampling as sampling
# import diffuser.utils as utils
# from diffuser.models.temporal import ValueFunction
# from diffuser.sampling.guides import ValueGuide

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# #-----------------------------------------------------------------------------#
# #----------------------------------- setup -----------------------------------#
# #-----------------------------------------------------------------------------#

# class Parser(utils.Parser):
#     # dataset: str = 'walker2d-medium-replay-v2'
#     dataset: str = 'hopper-medium-expert-v2'
#     config: str = 'config.locomotion'
#     method: str = 'cfm'
#     loadbase: str = 'logs'
#     preprocess_fns: list = []
    
#     # cov-g 관련 인자 추가
#     guidance_type: str = 'direct'  # 'direct', 'use_learned_v', 'rw' 중 선택
#     guidance_scale: float = 1.0    # 가이딩 강도 조절
#     value_model_path: str = None   # 가치 함수 모델 경로 (없으면 새로 생성)
#     value_hidden_dim: int = 256    # 가치 함수 모델의 hidden dimension


# args = Parser().parse_args('plan')

# args.batch_size = 1
# #-----------------------------------------------------------------------------#
# #---------------------------------- loading ----------------------------------#
# #-----------------------------------------------------------------------------#

# ## load diffusion model from disk
# diffusion_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.diffusion_loadpath,
#     epoch=args.diffusion_epoch, seed=args.seed,
# )

# diffusion = diffusion_experiment.ema
# dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer

# # 가치 함수 모델 생성 및 초기화
# value_model = ValueFunction(
#     horizon=diffusion.horizon,
#     transition_dim=diffusion.transition_dim,
#     cond_dim=args.value_hidden_dim
# )

# # 가치 함수 모델 로드 (경로가 제공된 경우)
# if args.value_model_path and os.path.exists(args.value_model_path):
#     value_model.load_state_dict(torch.load(args.value_model_path))
#     print(f"Loaded value model from {args.value_model_path}")
# else:
#     print("Using untrained value model")

# # 가치 함수 가이드 생성
# value_guide = ValueGuide(model=value_model, verbose=False)

# # cov-g 방법으로 리워드 가이딩 활성화
# diffusion.enable_guidance(
#     value_model=value_model,
#     guidance_type=args.guidance_type,
#     scale=args.guidance_scale
# )

# logger_config = utils.Config(
#     utils.Logger,
#     renderer=renderer,
#     logpath=args.savepath,
#     vis_freq=args.vis_freq,
#     max_render=args.max_render,
# )  

# # GuidedPolicy 클래스 사용 (수정된 버전)
# from diffuser.sampling.policies import GuidedPolicy

# policy_config = utils.Config(
#     GuidedPolicy,
#     diffusion_model=diffusion,
#     normalizer=dataset.normalizer,
#     preprocess_fns=args.preprocess_fns,
#     args=args
# )

# logger = logger_config()
# policy = policy_config()

# #-----------------------------------------------------------------------------#
# #--------------------------------- main loop ---------------------------------#
# #-----------------------------------------------------------------------------#

# env = dataset.env
# comp_time = []
# safety = []
# scores = []
# import time
# for kk in range(10):

#     observation = env.reset()

#     ## observations for rendering
#     rollout = [observation.copy()]
#     total_reward = 0
#     for t in range(args.max_episode_length):
        
#         # if t % 10 == 0: print(args.savepath, flush=True)

#         ## save state for rendering only
#         state = env.state_vector().copy()

#         ## format current observation for conditioning
#         conditions = {0: observation}

#         start = time.time()
#         action, trajectories, diffusion_paths, sum_elbo = policy(conditions, batch_size=args.batch_size, verbose=(t==0))
#         end = time.time()
        
#         # 안전성 값 (b_min) 대신 sum_elbo 사용 (또는 다른 메트릭으로 대체)
#         if t == 0:
#             safety.append(sum_elbo)
#             comp_time.append(end-start)
            
#         ## execute action in environment
#         next_observation, reward, terminal, _ = env.step(action)

#         ## print reward and score
#         total_reward += reward
#         score = env.get_normalized_score(total_reward)
#         print(
#             f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
#             f'guidance_type: {args.guidance_type} | scale: {args.guidance_scale}',
#             flush=True,
#         )

#         ## update rollout observations
#         rollout.append(next_observation.copy())

#         ## render every `args.vis_freq` steps
#         # logger.log(t, samples, state, rollout, diffusion)

#         if terminal:
#             break

#         observation = next_observation
#     scores.append(score)

# ## write results to json file at `args.savepath`
# # logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
# import numpy as np
# comp_time = np.array(comp_time)
# safety = np.array(safety)
# scores = np.array(scores)

# print("safety/elbo: ", np.min(safety))
# print("score mean: ", np.mean(scores))
# print("score std: ", np.std(scores))
# print("computation time: ", np.mean(comp_time))
# print(f"Guidance type: {args.guidance_type}, Guidance scale: {args.guidance_scale}")


# run script is under.
# python scripts/plan_guided_v2.py --dataset hopper-medium-expert-v2 --logbase logs --guidance_type rw --guidance_scale 1.0
# xvfb-run -s "-screen 0 1400x900x24" python scripts/plan_guided_v2.py     --dataset hopper-medium-expert-v2 --guidance_type rw --guidance_scale 2.0
# xvfb-run -s "-screen 0 1400x900x24" python scripts/plan_guided_v2.py     --dataset walker2d-medium-expert-v2 --guidance_type rw --guidance_scale 2.0

import pdb
import os
import torch
import diffuser.sampling as sampling
import diffuser.utils as utils
from gym.wrappers.monitor import Monitor
import imageio
from diffuser.models.temporal import ValueFunction
from diffuser.sampling.guides import ValueGuide

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    
    # cov-g 관련 인자 추가
    guidance_type: str = 'direct'  # 'direct', 'use_learned_v', 'rw' 중 선택
    guidance_scale: float = 1.0    # 가이딩 강도 조절
    value_model_path: str = None   # 가치 함수 모델 경로 (없으면 새로 생성)
    value_hidden_dim: int = 256    # 가치 함수 모델의 hidden dimension


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 가치 함수 모델 생성 및 초기화
value_model = ValueFunction(
    horizon=diffusion.horizon,
    transition_dim=diffusion.transition_dim,
    cond_dim=args.value_hidden_dim
).to(device)

# 가치 함수 모델 로드 (경로가 제공된 경우)
if args.value_model_path and os.path.exists(args.value_model_path):
    value_model.load_state_dict(torch.load(args.value_model_path))
    print(f"Loaded value model from {args.value_model_path}")
else:
    print("Using untrained value model")

# 가치 함수 가이드 생성
value_guide = ValueGuide(model=value_model).to(device)

# cov-g 방법으로 리워드 가이딩 활성화
diffusion.enable_guidance(
    value_model=value_model,
    guidance_type=args.guidance_type,
    scale=args.guidance_scale
)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)  

# GuidedPolicy 클래스 사용 (수정된 버전)
from diffuser.sampling.policies import GuidedPolicy

policy_config = utils.Config(
    GuidedPolicy,
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
env = Monitor(env,
              "videos/",
              video_callable=lambda ep_id: True,
              force=True)
comp_time = []
safety = []
scores = []
import time
for kk in range(1):

    observation = env.reset()
    frames = [env.render(mode='rgb_array')] 
    ## observations for rendering
    rollout = [observation.copy()]
    total_reward = 0
    for t in range(args.max_episode_length): #max = 1000
        
        # if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}

        start = time.time()
        action, trajectories, diffusion_paths, sum_elbo = policy(conditions, batch_size=args.batch_size, verbose=(t==0))
        end = time.time()
        
        # 안전성 값 (b_min) 대신 sum_elbo 사용 (또는 다른 메트릭으로 대체)
        if t == 0:
            safety.append(sum_elbo)
            comp_time.append(end-start)
            
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        frames.append(env.render(mode='rgb_array'))  

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'guidance_type: {args.guidance_type} | scale: {args.guidance_scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples, state, rollout, diffusion)

        if terminal:
            break

        observation = next_observation
    scores.append(score)
    env.close()

    imageio.mimsave(f'videos/{args.dataset}_rollout.mp4', frames, fps=30)

## write results to json file at `args.savepath`
# logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
import numpy as np
comp_time = np.array(comp_time)
safety = np.array(safety)
scores = np.array(scores)

print("safety/elbo: ", np.min(safety))
print("score mean: ", np.mean(scores))
print("score std: ", np.std(scores))
print("computation time: ", np.mean(comp_time))
print(f"Guidance type: {args.guidance_type}, Guidance scale: {args.guidance_scale}")

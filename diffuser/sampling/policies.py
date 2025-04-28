# from collections import namedtuple
# import torch
# import einops
# import pdb

# import diffuser.utils as utils
# from diffuser.datasets.preprocessing import get_policy_preprocess_fn
# from diffuser.models.temporal import ValueFunction


# Trajectories = namedtuple('Trajectories', 'actions observations values')

# class GuidedPolicy:

#     def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, args, **sample_kwargs):
#         self.guide = guide
#         self.diffusion_model = diffusion_model
#         self.normalizer = normalizer
#         self.action_dim = diffusion_model.action_dim
#         self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
#         self.sample_kwargs = sample_kwargs

#             # 리워드 가이딩 활성화 (args에서 설정 가져오기)
#         if hasattr(args, 'guidance_enabled') and args.guidance_enabled:  # 가치 함수 모델 가져오기
            
#             # 가치 함수 모델 생성
#             value_model = ValueFunction(
#                 horizon=diffusion_model.horizon,
#                 transition_dim=diffusion_model.transition_dim,
#                 hidden_dim=args.value_hidden_dim
#             ).to(diffusion_model.device)
            
#             # 가치 함수 모델 로드 (args에서 경로 가져오기)
#             if hasattr(args, 'value_model_path') and args.value_model_path:
#                 value_model.load_state_dict(torch.load(args.value_model_path))
            
#             # 리워드 가이딩 활성화
#             diffusion_model.enable_guidance(
#                 value_model=value_model,
#                 guidance_type=args.guidance_type,
#                 scale=args.guidance_scale
#             )

#     def __call__(self, conditions, batch_size=1, verbose=True):
#         conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
#         conditions = self._format_conditions(conditions, batch_size)

#         ## run reverse diffusion process
#         samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
#         trajectories = utils.to_np(samples.trajectories)

#         ## extract action [ batch_size x horizon x transition_dim ]
#         actions = trajectories[:, :, :self.action_dim]
#         actions = self.normalizer.unnormalize(actions, 'actions')

#         ## extract first action
#         action = actions[0, 0]

#         normed_observations = trajectories[:, :, self.action_dim:]
#         observations = self.normalizer.unnormalize(normed_observations, 'observations')

#         trajectories = Trajectories(actions, observations, samples.values)
#         return action, trajectories

#     @property
#     def device(self):
#         parameters = list(self.diffusion_model.parameters())
#         return parameters[0].device

#     def _format_conditions(self, conditions, batch_size):
#         conditions = utils.apply_dict(
#             self.normalizer.normalize,
#             conditions,
#             'observations',
#         )
#         conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
#         conditions = utils.apply_dict(
#             einops.repeat,
#             conditions,
#             'd -> repeat d', repeat=batch_size,
#         )
#         return conditions



import torch
import torch.nn as nn
import numpy as np
from diffuser.models.guidance_matcher import GuidanceMatcher
from diffuser.sampling.guides import ValueGuide

class GuidedPolicy:
    """
    수정된 GuidedPolicy 클래스: guide 매개변수를 직접 전달하지 않고 
    CFM 모델의 enable_guidance 메서드를 통해 가이딩을 적용합니다.
    """
    def __init__(self, diffusion_model, normalizer, guide=None, preprocess_fns=None, args=None, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.preprocess_fns = preprocess_fns or []
        self.args = args
        self.sample_kwargs = sample_kwargs
        
        # guide 객체가 전달된 경우, 이를 사용하여 diffusion_model에 가이딩 활성화
        if guide is not None and hasattr(guide, 'model') and not self.diffusion_model.guidance_enabled:
            self.diffusion_model.enable_guidance(
                value_model=guide.model,
                guidance_type=getattr(args, 'guidance_type', 'direct'),
                scale=getattr(args, 'guidance_scale', 1.0)
            )
            print(f"Enabled guidance with type: {getattr(args, 'guidance_type', 'direct')}, scale: {getattr(args, 'guidance_scale', 1.0)}")
        
        # 가이딩 관련 속성 저장
        self.guidance_enabled = self.diffusion_model.guidance_enabled
        self.guidance_type = getattr(args, 'guidance_type', 'direct') if self.guidance_enabled else None
        self.guidance_scale = getattr(args, 'guidance_scale', 1.0) if self.guidance_enabled else None

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _preprocess_observation(self, observation):
        for fn in self.preprocess_fns:
            observation = fn(observation)
        return observation

    def _format_conditions(self, conditions, batch_size):
        """
        조건을 정규화하고 배치 크기에 맞게 포맷팅합니다.
        """
        from diffuser.utils import apply_dict, to_torch, to_np
        import einops
        
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, batch_size=1, verbose=False):
        """
        정책 호출 메서드: 조건에 따라 액션을 생성합니다.
        guide 매개변수를 전달하지 않고 diffusion_model을 직접 호출합니다.
        """
        from diffuser.utils import to_np, to_torch
        
        # 조건 포맷팅
        conditions = self._format_conditions(conditions, batch_size)
        
        # diffusion 모델 호출 (guide 매개변수 없이)
        sample, diffusion = self.diffusion_model(conditions)
        
        # 결과 처리
        sample = to_np(sample)
        diffusion = to_np(diffusion)
        
        # 액션 추출 및 정규화 해제
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        action = actions[0, 0]  # 첫 번째 액션
        
        # 관측 추출 및 정규화 해제
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        # diffusion 경로 처리
        normed_diffusion = diffusion[:,:,:,self.action_dim:]
        diffusions = self.normalizer.unnormalize(normed_diffusion, 'observations')
        
        # 디버그 정보 출력
        if verbose:
            print(f"Guidance enabled: {self.guidance_enabled}")
            if self.guidance_enabled:
                print(f"Guidance type: {self.guidance_type}")
                print(f"Guidance scale: {self.guidance_scale}")
        
        # 결과 반환 (sum_elbo는 0으로 설정)
        from collections import namedtuple
        Trajectories = namedtuple('Trajectories', 'actions observations')
        trajectories = Trajectories(actions, observations)
        sum_elbo = 0
        
        return action, trajectories, diffusions, sum_elbo

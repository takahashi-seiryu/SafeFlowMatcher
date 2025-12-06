import torch
import torch.nn as nn
import numpy as np
import pdb
from diffuser.models.guidance_matcher import GuidanceMatcher
from diffuser.sampling.guides import ValueGuide

class GuidedPolicy:
    """
    Modified GuidedPolicy: applies guidance via the CFM model's enable_guidance
    method instead of passing the guide parameter directly.
    """
    def __init__(self, diffusion_model, normalizer, guide=None, preprocess_fns=None, args=None, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.preprocess_fns = preprocess_fns or []
        self.args = args
        self.sample_kwargs = sample_kwargs
        
        # if a guide object is provided, use it to enable guidance on the diffusion model
        if guide is not None and hasattr(guide, 'model') and not self.diffusion_model.guidance_enabled:
            self.diffusion_model.enable_guidance(
                value_model=guide.model,
                guidance_type=getattr(args, 'guidance_type', 'direct'),
                scale=getattr(args, 'guidance_scale', 1.0)
            )
            print(f"Enabled guidance with type: {getattr(args, 'guidance_type', 'direct')}, scale: {getattr(args, 'guidance_scale', 1.0)}")
        
        # store guidance-related attributes
        self.guidance_enabled = self.diffusion_model.guidance_enabled
        self.guidance_type = getattr(args, 'guidance_type', 'direct') if self.guidance_enabled else None
        self.guidance_scale = getattr(args, 'guidance_scale', 1.0) if self.guidance_enabled else None

    @property
    def device(self):
        # parameters = list(self.diffusion_model.parameters())
        # return parameters[0].device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _preprocess_observation(self, observation):
        for fn in self.preprocess_fns:
            observation = fn(observation)
        return observation

    def _format_conditions(self, conditions, batch_size):
        """
        Normalize conditions and format them to match the batch size.
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
        Policy call: generate an action conditioned on input.
        Calls diffusion_model directly without passing the guide parameter.
        """
        from diffuser.utils import to_np, to_torch
        
        # format conditions
        conditions = self._format_conditions(conditions, batch_size)
        _min, _max = self.normalizer.normalizers['observations'].min_max()
        self.diffusion_model.min = _min
        self.diffusion_model.max = _max
        # call diffusion model (without the guide parameter)
        sample, diffusion, b_min = self.diffusion_model(conditions)
        
        # process results
        sample = to_np(sample)
        diffusion = to_np(diffusion)
        
        # extract actions and unnormalize
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        action = actions[0, 0]  # first action
        
        # extract observations and unnormalize
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        # process diffusion paths
        normed_diffusion = diffusion[:,:,:,self.action_dim:]
        diffusions = self.normalizer.unnormalize(normed_diffusion, 'observations')
        
        # # debug output
        # if verbose:
        #     print(f"Guidance enabled: {self.guidance_enabled}")
        #     if self.guidance_enabled:
        #         print(f"Guidance type: {self.guidance_type}")
        #         print(f"Guidance scale: {self.guidance_scale}")
        
        # return results (sum_elbo is set to 0)
        from collections import namedtuple
        Trajectories = namedtuple('Trajectories', 'actions observations values')
        trajectories = Trajectories(actions, observations, None)
        
        return action, trajectories, diffusions, b_min

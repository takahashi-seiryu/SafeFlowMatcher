import socket
from diffuser.utils import watch

#------------------------ watched args ------------------------#
diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('normalizer', ''),
    ('batch_size', 'b'),
]

#------------------------ base config ------------------------#

base = {
    'diffusion': {
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['locomotion_preprocess_fn'],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'n_samples': 2,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 32,
        'device': 'cuda',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'normalizer': 'LimitsNormalizer',
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'default',
        'conditional': False,
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    }
}

#------------------------ cfm config ------------------------#

cfm = {
    'diffusion': {
        'model': 'models.TemporalUnet',
        'diffusion': 'models.CFM',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['locomotion_preprocess_fn'],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        'logbase': 'logs',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),

        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'n_samples': 2,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 32,
        'device': 'cuda',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'normalizer': 'LimitsNormalizer',
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'cfm',
        'conditional': False,
        'diffusion_loadpath': 'f:cfm/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    }
}

#------------------------ overrides ------------------------#

hopper_medium_v2 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 100,
        'preprocess_fns': ['locomotion_preprocess_fn'],
        'batch_size': 64,  
    },
    'plan': {
        'horizon': 32,
        'n_diffusion_steps': 100,
        'batch_size': 64,  # planning시 샘플링할 trajectory 수
    },
}

halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 100,
        'batch_size': 64,
    },
    'plan': {
        'horizon': 32,
        'n_diffusion_steps': 100,
        'batch_size': 32,
    },
}

walker2d_medium_v2 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 100,
    },
    'plan': {
        'horizon': 32,
        'n_diffusion_steps': 100,
    },
}

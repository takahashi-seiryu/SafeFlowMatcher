import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args


diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]


args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

# logbase = '/home/janner/mount/logs/diffuser-singularity/logs'
logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 600,  #32
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        # 'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 600,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 500,  #1000
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 600,    #32 ->600
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,  # 0.997 ->0.99

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '40',
    },
}


cfm = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet', # TemporalUnet, ConditionalUnet1D (Not implemented yet)
        'diffusion': 'models.CFM',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 25000,#10000
        'loss_type': 'l2',
        'n_train_steps': 5e5, #2e6
        'batch_size': 256,  # prev is 32
        'learning_rate': 2e-4, # prev is 2e-4
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 1000, # 1000
        'sample_freq': 5000, # prev is 1000
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## cfm model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '58',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:cfm/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 600,  #4
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 600,   #4
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 600,    #4
        # 'n_guide_steps': 1,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}

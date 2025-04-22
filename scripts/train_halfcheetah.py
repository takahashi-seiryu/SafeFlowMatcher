from diffuser.utils import watch
import diffuser.utils as utils
from copy import deepcopy
import torch
import torch.nn as nn
import pdb
import os
from diffuser.models.temporal_mlp import TemporalMLP

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'

    # Required paths
    logbase: str = 'logs'
    savepath: str = 'logs/halfcheetah'  # Set default savepath

    # Training
    n_train_steps: int = int(1e6)
    n_steps_per_epoch: int = 10000
    n_saves: int = 5
    bucket: str = None
    save_parallel: bool = False  # Single GPU 설정

    # Model
    n_diffusion_steps: int = 100
    horizon: int = 32
    n_reference: int = 8
    n_samples: int = 8
    hidden_dim: int = 128
    time_embed_dim: int = 128
    n_hidden: int = 4
    action_weight: float = 1
    loss_weights: str = None
    loss_discount: float = 1

args = Parser().parse_args('diffusion')

# Create save directory if it doesn't exist
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    TemporalMLP,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    time_embed_dim=args.time_embed_dim,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    hidden_dim=args.hidden_dim,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    num_workers=4,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()  # Single GPU 사용
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

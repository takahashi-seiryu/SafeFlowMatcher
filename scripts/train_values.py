import diffuser.utils as utils
import pdb
import torch
import numpy as np

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # dataset: str = 'walker2d-medium-replay-v2'
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'
    seed: int = 42

args = Parser().parse_args('values')


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
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
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
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,  # 실제 데이터 차원에 맞춤
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type='value_l1',
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
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

print(f"[디버그] observation_dim: {observation_dim}")
print(f"[디버그] action_dim: {action_dim}")
print(f"[디버그] transition_dim: {observation_dim + action_dim}")
print(f"[디버그] dataset horizon: {dataset.horizon}")
print(f"[디버그] diffusion horizon: {diffusion.horizon}")
print(f"[디버그] dataset[0] trajectories shape: {dataset[0].trajectories.shape}")


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)

data = dataset[0]

device = next(diffusion.parameters()).device  # Get the device of the model
trajectories = torch.as_tensor(data.trajectories).unsqueeze(0).to(device)  # (1, 600, 14)
conditions = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in data.conditions.items()}  # dict
target = torch.as_tensor(data.values).unsqueeze(0).to(device)  # (1, 1)

batch = (trajectories, conditions, target)

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

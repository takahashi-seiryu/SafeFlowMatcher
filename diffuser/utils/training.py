import os
import copy
import numpy as np
import torch
import einops
import pdb
import torch.nn as nn

from contextlib import contextmanager

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class _EMAModelView(nn.Module):
    """
    Proxy that runs trainer.model with EMA weights temporarily.
    Mimics the ema_model interface expected by older code as closely as possible.
    """
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer  # circular reference is fine

    # nn.Module-compatible call (e.g., ema_model(x))
    def forward(self, *args, **kwargs):
        with self._trainer.use_ema_weights():
            return self._trainer.model(*args, **kwargs)

    # Wrap frequently used sampling utilities (add only what is needed)
    def conditional_sample(self, *args, **kwargs):
        with self._trainer.use_ema_weights():
            return self._trainer.model.conditional_sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._trainer.use_ema_weights():
            return self._trainer.model.sample(*args, **kwargs)

    # Delegate all other attributes/methods to the underlying model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._trainer.model, name)

    # Convenience for chaining .eval() / .train()
    def eval(self):
        self._trainer.model.eval()
        return self

    def train(self, mode: bool = True):
        self._trainer.model.train(mode)
        return self

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.shadow = None

    def init_from(self, state_dict: dict):
        # track only float tensors (int, bool, etc. do not need averaging)
        self.shadow = {
            k: v.detach().clone()
            for k, v in state_dict.items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, state_dict: dict):
        if self.shadow is None:
            self.init_from(state_dict)
            return
        for k, v in state_dict.items():
            if not torch.is_floating_point(v):
                continue
            self.shadow[k].mul_(self.beta).add_(v.detach(), alpha=1.0 - self.beta)

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self._ema_view = _EMAModelView(self)  # added
        self.ema_model = self._ema_view    # added
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        # self.ema_model.load_state_dict(self.model.state_dict())
        self.ema.init_from(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        #     return
        # self.ema.update_model_average(self.ema_model, self.model)
        else:
            self.ema.update(self.model.state_dict())


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)

            # if self.step == 0 and self.sample_freq: 
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     self.render_samples()

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            # 'ema': self.ema_model.state_dict()
            'ema' : self.ema.shadow, 
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema.shadow = data.get('ema', None)
        # self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

    @contextmanager
    def use_ema_weights(self):
        """
        Temporarily load EMA weights for inference.
        - In grad mode, keep the weights until backward finishes to avoid version-bump autograd errors.
        - In no_grad mode (pure inference), restore the original weights after use.
        """
        if self.ema.shadow is None:
            yield
            return

        restore_after = not torch.is_grad_enabled()  # restore only when backward is not needed
        orig = None
        if restore_after:
            # back up only when restoration is required
            orig = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # swap in EMA weights (no_grad prevents leaving traces in the computation graph)
        with torch.no_grad():
            tmp = self.model.state_dict()
            for k, v in self.ema.shadow.items():
                if k in tmp and torch.is_floating_point(tmp[k]):
                    tmp[k].copy_(v)

        try:
            yield
        finally:
            if restore_after and orig is not None:
                with torch.no_grad():
                    self.model.load_state_dict(orig, strict=False)

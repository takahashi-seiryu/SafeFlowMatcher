# SafeFlowMatcher

### Safe and Fast Planning using Flow Matching with Control Barrier Functions

**ICLR 2026**

[Jeongyong Yang](https://github.com/takahashi-seiryu)\*, [Seunghwan Jang](https://github.com/Jang-seunghwan)\*†, SooJean Han

Korea Advanced Institute of Science and Technology (KAIST)

\*Equal contribution. †Corresponding author.

[[Paper]](https://openreview.net/forum?id=refcXHU1Nh) [[Project Page]](https://takahashi-seiryu.github.io/SafeFlowMatcher/)

<!-- Add a method overview figure here -->
<!-- <p align="center">
  <img src="assets/overview.png" width="80%">
</p> -->

---

## Installation

```bash
conda env create -f environment.yml
conda activate safe_cfm
pip install -e .
pip install qpth cvxpy cvxopt
pip install torchdyn torchdiffeq torchcfm
pip install git+https://github.com/atong01/conditional-flow-matching.git
```

## Training

### Maze2D

**Conditional Flow Matching:**
```bash
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1 --method cfm
```

**Diffuser (baseline):**
```bash
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1 --method base
```

### Locomotion
```bash
python scripts/train.py --dataset walker2d-medium-expert-v2
```

The default hyperparameters are listed in `config/maze2d.py` and `config/locomotion.py`. You can override any of them with flags, e.g., `--n_diffusion_steps 100`.

### Value Function
```bash
python scripts/train_values.py --dataset walker2d-medium-expert-v2
```

## Planning (Evaluation)

### Maze2D

**Conditional Flow Matching:**
```bash
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm
```

**Diffuser (baseline):**
```bash
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method base
```

### Locomotion
```bash
python scripts/plan_guided.py --dataset walker2d-medium-expert-v2 --logbase logs
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yang2026safeflowmatcher,
  title={SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions},
  author={Yang, Jeongyong and Jang, Seunghwan and Han, SooJean},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgements

- The diffusion model implementation is based on [Diffuser](https://github.com/jannerm/diffuser) by Michael Janner.
- The safe diffusion implementation is based on [SafeDiffuser](https://github.com/Weixy21/SafeDiffuser) by Wei Xiao.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

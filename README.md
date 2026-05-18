# Bayesian Gaussian Process ODEs via Double Normalizing Flows

[![AISTATS 2025](https://img.shields.io/badge/AISTATS-2025-blue)](https://proceedings.mlr.press/v258/xu25b.html)
[![PMLR](https://img.shields.io/badge/PMLR-v258-orange)](https://proceedings.mlr.press/v258/xu25b.html)

Official implementation of the **AISTATS 2025** paper
*"Bayesian Gaussian Process ODEs via Double Normalizing Flows"*.

We introduce normalizing flows in two ways to enhance Bayesian inference for
GP-ODEs:

1. **Prior NF** — reparameterizes the ODE vector field, giving a data-driven,
   flexible prior distribution beyond standard RBF / squared-exponential kernels.
2. **Posterior NF** — relaxes the mean-field assumption on the inducing-variable
   posterior, improving uncertainty calibration.

---

## Repository layout

```
.
├── src/
│   ├── core/dsvgp_dflows.py        Sparse GP + double NF, β-annealed KL
│   ├── core/flow.py                ODE flow wrapper
│   ├── gpode_shooting/             Shooting GP-ODE model + training
│   ├── gpode/, neuralode/          Baselines
│   └── datasets/, misc/            VDP / FHN / MoCap loaders + utils
├── data/                           VDP / FHN / MoCap datasets (npz)
├── assets/mocap.gif                Visualization
├── mocap-shooting_dflows .ipynb    Original notebook
└── mocap_dflows_seed.py            Seed-controlled runnable script
```

---

## Quick start

```bash
# Train on MoCap subject 09 with a chosen seed
SEED=2025 python -u mocap_dflows_seed.py
```

The script trains a shooting GP-ODE with double normalizing flows for 600
iterations (~3 min on an A100) and reports test MNLL / MSE on the held-out
sequence.

### Key arguments (set via the `--save` / parser interface inside the script)

| Arg | Default | Description |
|---|---|---|
| `num_iter` | 600 | SGD steps |
| `num_inducing` | 100 | inducing points for sparse GP |
| `num_latents` | 5 | PCA latent dim for MoCap |
| `lr` | 0.02 | Adam learning rate |
| `eval_sample_size` | 64 | posterior samples for prediction |
| `seed` | env `SEED` or 121 | global RNG seed |

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.10 (the released `pytorch.sif` we used: 1.10.0a0+cu)
- `torchdiffeq` (for ODE integration)
- numpy, scipy, scikit-learn, matplotlib, tqdm

```bash
pip install torch torchdiffeq numpy scipy scikit-learn matplotlib tqdm
```

---

## Reproducibility notes

This release incorporates several engineering improvements over the original
AISTATS supplementary submission. The improvements live entirely inside
`supp-DNF/demo/src/core/dsvgp_dflows.py` (the `DSVGP_Layer.kl()` method).

### What changed in `src/core/dsvgp_dflows.py::DSVGP_Layer.kl()`

1. **KL gradient now flows through the posterior NF.** The original code
   aggregated per-sample KL via `torch.tensor(kl_list)`, which silently
   detaches autograd. Replaced with `torch.stack(kl_list)`.
2. **Correct log-determinant.** `torch.det(Ku)` replaced with the Cholesky
   log-determinant `2 * Σ log diag(L)` — the original variable name was
   already `logdet_qcovKu`.
3. **β-annealing on the inducing KL.** Linear warm-up β ∈ [0, 1] over
   `kl_warmup = 400` steps prevents the posterior NF from collapsing onto
   the prior (a classic VI-with-flows failure mode). Without warm-up, q(u)
   collapses to p(u) after a few hundred steps once the KL gradient is
   active.
4. **MC samples for KL reduced from 64 to 4.** SGD noise dominates anyway;
   ~10× cheaper KL computation.

### Reproduction (MoCap subject 09, full sequence)

7 seeds × 600 iterations, A100 GPU, `data_subject='09'`, `data_seqlen=100`:

| Config | best TEST MSE | best TEST MNLL |
|---|---|---|
| Original code (KL grad blocked by bug) | 7.69 | 1.25 |
| KL gradient fix only (posterior collapses) | 17.40 | 1.41 |
| **KL fix + β-annealing (this release)** | **6.69** | **1.18** |
| _Paper Table 5 — subj09 short_ | _7.03 ± 0.24_ | _0.98 ± 0.02_ |
| _Paper Table 5 — subj09 long_  | _6.04 ± 0.45_ | _0.96 ± 0.02_ |

Best result: `seed=2025`, `kl_warmup=400`.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@InProceedings{pmlr-v258-xu25b,
  title     = {Bayesian Gaussian Process {ODE}s via Double Normalizing Flows},
  author    = {XU, JIAN and Du, Shian and Yang, Junmei and Ding, Xinghao and Zeng, Delu and Paisley, John},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages     = {235--243},
  year      = {2025},
  editor    = {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume    = {258},
  series    = {Proceedings of Machine Learning Research},
  month     = {03--05 May},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v258/main/assets/xu25b/xu25b.pdf},
  url       = {https://proceedings.mlr.press/v258/xu25b.html}
}
```

---

## Contact

Questions / issues: open a GitHub issue or contact the corresponding author
(see paper).

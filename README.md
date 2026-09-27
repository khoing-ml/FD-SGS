# FD-SGS: Z-Image-Turbo, 8 effective steps

Experimental implementation of [the local plan](plans/fd_sgs_improvement.md), inspired by
[SGS](https://github.com/NhuGiap04/SGS). This is a standalone Z-Image adapter, not a copy
of the upstream SD/SDXL pipelines. It does not train or modify model weights.

## Install and run

Use Python 3.10–3.13 with a CUDA-compatible PyTorch installation. Python 3.12 is a good
default. The first real run downloads Z-Image-Turbo and PickScore model weights.

For runtime installation from the repository root, use `python -m pip install -r requirements.txt`.
This installs the project and its dependencies from `pyproject.toml`, including the `fd-sgs` CLI.
The command below also includes test dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

fd-sgs --prompt "A red panda wearing a tiny astronaut helmet, cinematic photography" \
  --method compare --particles 4 --offload --output outputs/panda-seed42
```

`compare` runs unguided/Best-of-K, independent FD, and FD-SGS with the same initial
anchor noise and probe-noise stream. Each method saves all particles, `best.png`, and
`metrics.json`; the parent directory gets `comparison.json`. Output directories must
be empty to prevent accidental overwrites. Use a new output path for each experiment.

## SGS prompts and reward evaluation

The project bundles the two files from
[SGS text2img/prompts](https://github.com/NhuGiap04/SGS/tree/main/text2img/prompts)
at revision `f410ab526c996bb9e7828ba92f02b68e35473a2f`:

| `--prompt-dataset` | File | Number of prompts |
|---|---|---:|
| `image-reward` | `benchmark_ir.json` | 100 |
| `hpsv2` | `hps_v2_all_eval.txt` | 50 |

Hashes and source paths are in [fd_sgs/data/prompts/source.json](fd_sgs/data/prompts/source.json).
Use `--prompts-file your_prompts.txt` for one prompt per nonblank line, or a JSON
list of strings or `{ "id": "...", "prompt": "..." }` records. `--start-index` and
`--max-prompts` select a stable slice; each prompt uses `seed + original index`,
so methods see identical initial noise for the same prompt. `--dry-run` prints
the selected prompts and settings without loading models or opening a W&B run.

Install all four reward backends and run a small prompt batch:

```bash
python -m pip install -r requirements-rewards.txt
fd-sgs --prompt-dataset image-reward --start-index 0 --max-prompts 2 \
  --method compare --reward pickscore --eval-rewards all \
  --particles 2 --height 512 --width 512 --offload \
  --wandb-mode offline --output outputs/sgs-batch-01
```

`--reward` chooses the **guidance and final-selection** scorer: `imagereward`,
`pickscore`, `hpsv2`, `clipscore`, `brightness`, or a `module:function` callable.
`--eval-rewards all` adds ImageReward, PickScore, HPSv2, and CLIPScore on **final
anchor images only**. You may list specific names instead. Already computed
guidance-reward final scores are reused; extra scorers do not change images or the
selected particle. All scores are higher-is-better and retain their own scales;
do not add or average unlike scores. Use `--hps-version v2.1` for the newer HPS
checkpoint; the default is v2.0, matching the SGS setup.

Batch outputs live under `prompt_00000/<method>/` and include candidate PNGs,
`best.png`, and `metrics.json` with every final scorer's per-particle values.
`results.csv` has one row per particle, prompt, and method; `batch_summary.json`
reports mean scores across all and guidance-selected particles for each method.
These are separate from oracle selection by an evaluation scorer. In batch mode,
W&B opens one run per prompt, grouped under the output name by default, with all
four final scorer tables and comparison charts. Model loading is reused across
prompts, and reward backends are loaded lazily and cached.

The ImageReward package needs the `openai-clip` dependency and a compatibility
shim for newer Transformers; the provided extra installs it. The HPSv2 backend
loads the official `xswu/HPSv2` weights into OpenCLIP ViT-H/14, avoiding the
HPSv2 package's old global-device behavior and test dependency pins. PickScore
uses its documented CLIP-H processor; CLIPScore uses OpenAI ViT-L/14 cosine
similarity as in SGS. Preprocessing differs somewhat from SGS's custom tensor
scorers, so treat scores as consistent within this runner, not numerically identical
to the upstream evaluation. Four reward backends plus Z-Image weights need
substantial CPU/GPU memory. For a quick setup check, use `--eval-rewards` only
after an initial run with the target reward alone.

For Colab, open [notebooks/fd_sgs_colab.ipynb](notebooks/fd_sgs_colab.ipynb).
Use **File → Upload notebook** in Google Colab if viewing this local workspace.
It accepts either a ZIP upload of this project or a GitHub clone, checks GPU
availability, installs dependencies, runs a small prompt batch, displays results,
and can copy outputs to Google Drive.

To run only FD-SGS:

```bash
fd-sgs --prompt "A mountain village reflected in a lake at sunrise" \
  --particles 4 --rho 0.025 --guidance-steps 1 2 3 4 5 6 \
  --rho-schedule linear-decay --rho-start-multiplier 2 \
  --twin-sampler fdfo --exploration 0.0025 --repulsion 0.1 \
  --offload --output outputs/village
```

To use Flow-GRPO Euler–Maruyama exploration with two probes per anchor:

```bash
fd-sgs --prompt "A mountain village reflected in a lake at sunrise" \
  --method compare --particles 4 --probes 2 \
  --twin-sampler flow-grpo --noise-level 0.05 \
  --offload --output outputs/village-flow-grpo
```

`--particles K` controls the number of final candidate images. `--probes B` controls
the number of local alternatives used to estimate each anchor's guidance; probes are
not returned as final candidates. The default is one FDFO probe per anchor.
For multiple probes, average their individually RMS-normalized, reward-weighted
directions before Stein aggregation. Each probe starts at its anchor's initial noise
and receives an independent exploration-noise draw at each step.

`--twin-sampler fdfo` uses the flow-adapted EDM overshoot/re-noise transition
described in the [FDFO reference implementation](https://github.com/NVlabs/finite-difference-flow-optimization).
The previous name `edm` remains an alias for the same transition. `--exploration`
sets its re-noising strength; the default is `0.0025`.

FDFO scores completed paired trajectories for its training update. To compare
finished images during inference-time guidance, use `--guidance-eval final-rollout`.
At each guided step, this deterministically finishes the current anchor and probe
states to sigma zero, decodes and scores those images, and uses their terminal latent
difference for the finite-difference direction. The normal sampler state is unchanged
by these lookaheads. The default `predicted-clean` mode scores one-step clean
estimates and is much cheaper. Future probe noise is omitted in the lookahead, so
`final-rollout` is a deterministic counterfactual, not a reproduction of FDFO training.
In W&B, guided previews are labeled `final_rollouts` in this mode.

`--twin-sampler flow-grpo` uses the `sde` drift and noise coefficients from the
[official Flow-GRPO sampler](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py).
It changes only the probe transitions; anchors still use deterministic Euler plus
the FD-Stein correction. At sigma=1, the noise denominator uses the second schedule
sigma, matching upstream. All eight probe transitions use the SDE formula, including
the last interval; final probes are discarded. Calculations use float32.
`--noise-level 0` gives deterministic Euler probes; `0.05` is an experimental starting
value, not a validated optimum for Z-Image. `--exploration` controls only FDFO probes,
while `--noise-level` controls only Flow-GRPO probes; the strengths are not interchangeable.
Neither parameter changes the anchor guidance strength.
This uses the sampling transition without GRPO training, transition log-probabilities,
DAS importance weights, tempering, or manifold projection. Existing Stein correction
normalization is retained for this initial pipeline experiment.

`--offload` moves pipeline components between CPU and CUDA. PickScore defaults to CPU
to leave GPU memory for generation; `--reward-device cuda` trades memory for speed.
VAE decoding is chunked one image at a time with slicing/tiling. The transformer still
evaluates all **K(1+B)** anchor/probe samples in one batch (2K by default). If memory is tight, start with
`--particles 2 --height 512 --width 512`; this is a plumbing test at a different
resolution, not an equivalent quality benchmark. Model CPU offload still needs enough
GPU memory for the active transformer and ample host RAM. No fixed VRAM claim is made.

Use `--reward brightness` to test image generation without downloading PickScore.
This optimizes brightness, not image quality. A custom black-box reward can be supplied
with `--reward my_module:score`, where `score(pil_images, prompt)` returns one finite
float per image, higher being better. Reward batches must not alter score meaning.

## Weights & Biases logging

W&B is included in the runtime dependencies. Logging defaults to disabled; select
`--wandb-mode online` to publish a run, or `--wandb-mode offline` to save logs locally.
One CLI invocation creates one W&B run, with separate method namespaces for comparisons.

```bash
python -m pip install -r requirements.txt
wandb login
fd-sgs --prompt "A mountain village reflected in a lake at sunrise" \
  --method compare --particles 4 --probes 2 --twin-sampler flow-grpo \
  --offload --output outputs/village-tracked \
  --wandb-mode online --wandb-project fd-sgs --wandb-name village-seed42 \
  --wandb-group zimage-8step --wandb-tags flow-grpo pickscore --wandb-previews
```

Use `--wandb-entity YOUR_TEAM` if needed. Authentication uses `wandb login` or
`WANDB_API_KEY`; credentials are not CLI arguments or part of the recorded config.
Online runs upload the prompt, configuration, metrics, and selected generated images.

The run includes:

- Full CLI settings, per-method sampling configs, seed, model/reward identifiers,
  and Torch/Diffusers/Transformers versions.
- Live per-method curves for sigma, anchor/probe reward mean/std/min/max,
  pairwise reward differences, individual anchor rewards, and correction ratios.
  Reward curves occur only at guided steps; final rewards are recorded separately.
- Final particle tables with every particle's score, selected images, and the best
  image; reward histograms, mean/best comparison charts, and deltas versus unguided.
- NFE, denoiser batch elements, reward counts, elapsed time, and peak CUDA memory.
- Per-method result artifacts containing `metrics.json` (including full reward traces)
  and `best.png`. All particle PNGs are also saved in the local output directory.

`--wandb-image-limit 8` caps final table images per method, always including the winner.
`--wandb-previews` also logs predicted-clean images at guided steps, capped separately
for anchors and probes. These are intermediate model predictions, not final samples.
They reuse existing reward decodes, so no extra denoiser or reward evaluations are made.
Image serialization adds logging time; `wall_seconds` includes live step logging but
excludes final W&B tables/artifacts and run shutdown. Compare latency with logging
disabled when benchmarking the sampler itself.

For a run without an account/network connection, replace `--wandb-mode online` with
`--wandb-mode offline` and skip `wandb login`. Logs live under `<output>/wandb/` and
can later be uploaded with `wandb sync <output>/wandb/offline-run-...`.
Failures finish the W&B run with a nonzero exit code. Logging errors surface rather
than silently discarding tracking; local generation results are saved before final
result logging.

## Implemented method

- Four anchors by default, each with one or more persistent stochastic probes sharing initial noise.
- Deterministic Euler anchor updates; probes use FDFO overshoot/re-noise by default,
  or Flow-GRPO Euler–Maruyama when selected. All probes for an anchor receive its Stein correction.
- At **one-based** steps 1 through 6, decode guidance endpoints. By default, the anchor
  prediction is `base - sigma_next * velocity` (equal to `x_sigma - sigma * velocity`);
  the probe prediction uses its freshly sampled endpoint at `sigma_next`. This gives
  step 1 a real finite-difference signal despite shared initial noise. Z-Image's
  transformer output is negated to obtain the scheduler's velocity convention. With
  `final-rollout`, score completed deterministic continuations instead.
- Decode both clean predictions, score with PickScore, and compute
  `(reward_twin - reward_anchor) * delta_clean_latent / (RMS(delta_clean_latent) + eps)`.
  With B probes, average B such directions per anchor; score each anchor only once per guided step.
- RBF interaction with the SGS median/log(K+1) bandwidth and analytic source-gradient
  repulsion. Independent FD uses the same estimator without kernel mixing or repulsion.
- Split correction after the base step, with per-particle correction/base norm ratio
  `min(rho_step, trust_ratio)`. By default, `rho_step` decays linearly over guided
  steps 1–6: 0.05, 0.045, 0.04, 0.035, 0.03, 0.025. The trust cap is 0.1.
  `--rho-schedule constant` restores a fixed `--rho` at every guided step;
  `--rho-start-multiplier` controls the initial/final ratio in linear mode.
  Zero fields stay zero.
- Return/rank only anchors. Unguided generation plus final ranking is Best-of-K.

**Eight-step convention:** Diffusers 0.36.0's native Z-Image pipeline sets `sigma_min=0`.
Its `num_inference_steps=9` schedule contains eight nonzero Euler intervals followed by
a zero-length interval at sigma zero. We preserve those eight intervals and skip the
redundant ninth transformer call. The unguided path is tested against the native
9-call pipeline on small synthetic components with the real scheduler. We do not
use the native `num_inference_steps=8` grid, which has only seven nonzero intervals.
See the [pinned pipeline source](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/z_image/pipeline_z_image.py).

PickScore uses its [documented CLIP-H processor](https://huggingface.co/yuvalkirstain/PickScore_v1)
and cosine similarity (SGS's score scale), without softmax or learned logit scaling.
Preprocessing follows the model card; this is not an exact reproduction of upstream
SGS's custom image resizing. Avoid comparing its absolute scores across implementations.

## Compute accounting and limitations

For K=4, B=1, and the default six guidance steps (either probe sampler):

| Method | Sequential NFE / transformer calls | Denoiser batch elements | Reward images |
|---|---:|---:|---:|
| Unguided / Best-of-4 | 8 | 32 | 4 final |
| Independent FD | 8 | 64 | 48 guidance + 4 final |
| FD-SGS | 8 | 64 | 48 guidance + 4 final |

For B probes, guided runs evaluate `8*K*(1+B)` denoiser batch elements and
`6*K*(1+B)+K` reward images with the default guidance schedule. For K=4, B=2,
that is 96 denoiser batch elements and 76 reward images. Sequential NFE remains eight.
Metrics record the active probe sampler, probe count, and total twin trajectories.
Step `twin_rewards` lists are flattened in probe-major order: all K anchors' first
probes, then all K anchors' second probes, and so on.

`final-rollout` adds `sum(8 - step for step in guidance_steps)` batched transformer
calls per guided method. For steps 1–6, that is **27 extra calls**, so a guided run
uses 35 total calls and `35*K*(1+B)` total denoiser batch elements; the number of
rewarded images remains the same. `sequential_nfe` and `denoiser_forward_calls`
continue to describe the eight main trajectory steps. Separate `rollout_*` and
`total_*` metrics account for the lookaheads in output JSON and W&B.

`reward_calls` counts adapter invocations (7 in guided runs), not internal microbatches
or remote requests. `reward_image_evaluations` counts individual scored images.
Metrics also record each guided step's scheduled strength, scores/correction ratios, sigmas, final scores,
best index, wall time, peak allocated/reserved CUDA memory, config, and library versions.
Time includes prompt encoding, sampling, VAE/reward work and final ranking, but excludes
model loading and image saving. CPU reward memory is not included in CUDA memory.

This is a testable research prototype, **not an established quality improvement**.
Predicted-clean reward reliability, latent endpoint directions applied directly to current
latents, and persistent twin locality remain empirical assumptions from the plan.
Shared randomness matches the noise inputs and compute budget; pairs naturally diverge
after the different methods steer their trajectories. Repulsion can still act when
reward differences vanish. For a complete no-guidance control use `unguided` or `--rho 0`.
Exact-gradient SGS, adaptive exploration, direction reuse, other kernels, full diversity
metrics, and 10-step backbones are not part of this first Z-Image experiment.

Suggested first sweep: fixed prompt/seed, `--rho 0.01`, `0.025`, `0.05`, `0.1`, then
repeat across several prompts/seeds. Reward improvement alone does not establish
visual quality or diversity.

## Tests

```bash
python -m pytest -q
python -m fd_sgs.cli --help
```

Tests need no downloaded model weights. They cover the FD direction, RBF source-gradient
sign against autograd, trust region, equivalence to the plan's exploration formula,
native pipeline parity, reproducibility, schedule, accounting, and invalid rewards.
They also cover Euler–Maruyama transition moments/endpoints, zero-noise ODE recovery,
multi-probe pairing/averaging, and CLI image/metrics output using synthetic components.
W&B tests verify disabled mode, failure cleanup, reward aggregation, image caps, and
a real offline SDK run with synthetic model components (including previews and comparison charts).
The Euler–Maruyama implementation was additionally compared against upstream Flow-GRPO
over the eight-step schedule (maximum absolute difference below 5e-7 on CPU float32).
Full Z-Image/PickScore generation requires the model downloads and suitable hardware.

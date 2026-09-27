"""Hyper-SD3 CFG-LoRA sampling on SD3's flow-matching schedule."""

from dataclasses import dataclass
import math
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .core import fd_direction, flow_grpo_step, relative_correction, rms, stein_field, stochastic_flow_step
from .hyper_sd import HYPER_LORAS


@dataclass(frozen=True)
class HyperSD3SamplingConfig:
    particles: int = 4
    steps: int = 16
    guidance_steps: tuple[int, ...] | None = None
    guidance_eval: str = "final-rollout"
    rho: float = 0.025
    rho_schedule: str = "linear-decay"
    rho_start_multiplier: float = 2.0
    repulsion: float = 0.1
    probes: int = 1
    trust_ratio: float = 0.1
    cfg_scale: float = 7.0
    twin_sampler: str = "flow-grpo"
    noise_level: float = 0.05
    exploration: float = 0.0025
    method: str = "fd-sgs"
    seed: int = 42
    height: int = 1024
    width: int = 1024
    decode_batch_size: int = 1
    unet_batch_size: int = 1
    backbone: str = "hyper-sd3"

    def __post_init__(self):
        if self.backbone != "hyper-sd3" or (self.backbone, self.steps) not in HYPER_LORAS:
            raise ValueError("Hyper-SD3 CFG LoRAs support 4, 8, or 16 steps")
        if self.guidance_steps is None:
            object.__setattr__(self, "guidance_steps", tuple(range(1, min(self.steps, 6) + 1)))
        if self.method not in {"fd-sgs", "independent-fd", "unguided"}:
            raise ValueError("Unknown sampling method")
        if self.guidance_eval != "final-rollout":
            raise ValueError("Hyper-SD3 guidance requires final-rollout reward evaluation")
        if self.twin_sampler not in {"flow-grpo", "fdfo"}:
            raise ValueError("Hyper-SD3 probes require flow-grpo or fdfo")
        if self.rho_schedule not in {"linear-decay", "constant"}:
            raise ValueError("Unknown rho schedule")
        if any(not isinstance(value, int) or value < 1 for value in
               (self.particles, self.probes, self.decode_batch_size, self.unet_batch_size)):
            raise ValueError("Particles, probes and batch sizes must be positive integers")
        if self.height < 16 or self.width < 16 or self.height % 8 or self.width % 8:
            raise ValueError("Height and width must be positive multiples of 8")
        if len(set(self.guidance_steps)) != len(self.guidance_steps) or any(
            s < 1 or s > self.steps for s in self.guidance_steps
        ):
            raise ValueError(f"Guidance steps must be unique one-based indices in 1..{self.steps}")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.rho, self.repulsion, self.trust_ratio, self.noise_level, self.exploration)):
            raise ValueError("Guidance and exploration strengths must be finite and nonnegative")
        if not math.isfinite(self.rho_start_multiplier) or self.rho_start_multiplier < 1:
            raise ValueError("rho_start_multiplier must be finite and at least 1")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale <= 1:
            raise ValueError("Hyper-SD3 CFG scale must be finite and greater than 1")


def decode_images(pipe, latents, batch_size):
    images = []
    for batch in latents.split(batch_size):
        z = batch.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        z = z + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(z, return_dict=False)[0]
        images.extend(pipe.image_processor.postprocess(decoded, output_type="pil"))
    return images


@torch.inference_mode()
def sample(pipe, prompt, reward, config=HyperSD3SamplingConfig(), *, on_step=None):
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift

    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError("Hyper-SD3 requires FlowMatchEulerDiscreteScheduler")
    if pipe.scheduler.config.get("stochastic_sampling", False) or pipe.scheduler.config.get("invert_sigmas", False):
        raise ValueError("Hyper-SD3 requires a deterministic, descending-sigma flow scheduler")
    device = torch.device(pipe._execution_device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    rng = torch.Generator(device=device).manual_seed(config.seed)
    probe_rng = torch.Generator(device=device).manual_seed(config.seed + 1)
    pos, neg, pooled_pos, pooled_neg = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=True)
    k = config.particles
    anchors = pipe.prepare_latents(k, pipe.transformer.config.in_channels, config.height,
                                   config.width, pos.dtype, device, rng, None)
    scheduler_kwargs = {}
    sc = pipe.scheduler.config
    if sc.get("use_dynamic_shifting", False):
        seq_len = ((anchors.shape[2] // pipe.transformer.config.patch_size)
                   * (anchors.shape[3] // pipe.transformer.config.patch_size))
        scheduler_kwargs["mu"] = calculate_shift(
            seq_len, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
            sc.get("base_shift", 0.5), sc.get("max_shift", 1.16))
    pipe.scheduler.set_timesteps(config.steps, device=device, **scheduler_kwargs)
    timesteps = list(pipe.scheduler.timesteps)
    sigmas = pipe.scheduler.sigmas
    if (len(timesteps) != config.steps or len(sigmas) != config.steps + 1 or
            float(sigmas[-1]) != 0 or not bool((sigmas[:-1] > sigmas[1:]).all())):
        raise ValueError("SD3 scheduler must produce descending nonzero intervals ending at sigma zero")
    guided = sorted(config.guidance_steps)
    rhos = {step: config.rho if config.rho_schedule == "constant" else
            config.rho * (1 + (config.rho_start_multiplier - 1) *
                          (1 - rank / max(len(guided) - 1, 1)))
            for rank, step in enumerate(guided)}
    stats = {"sequential_nfe": 0, "denoiser_forward_calls": 0, "denoiser_batch_elements": 0,
             "rollout_denoiser_forward_calls": 0, "rollout_denoiser_batch_elements": 0,
             "particles": k, "twin_trajectories": k * config.probes if config.method != "unguided" else 0,
             "probes_per_particle": config.probes if config.method != "unguided" else 0,
             "twin_sampler": config.twin_sampler if config.method != "unguided" else None,
             "reward_calls": 0, "reward_image_evaluations": 0,
             "guidance_reward_image_evaluations": 0, "steps": []}

    def predict(states, timestep, rollout=False):
        outputs = []
        for chunk in states.split(config.unet_batch_size):
            n = len(chunk)
            model_input = torch.cat([chunk.to(pipe.transformer.dtype)] * 2)
            embeds = torch.cat([neg.expand(n, -1, -1), pos.expand(n, -1, -1)])
            pooled = torch.cat([pooled_neg.expand(n, -1), pooled_pos.expand(n, -1)])
            model_t = timestep.expand(2 * n)
            output = pipe.transformer(hidden_states=model_input, timestep=model_t,
                                      encoder_hidden_states=embeds, pooled_projections=pooled,
                                      return_dict=False)[0]
            uncond, cond = output.chunk(2)
            outputs.append(uncond + config.cfg_scale * (cond - uncond))
            prefix = "rollout_" if rollout else ""
            stats[f"{prefix}denoiser_forward_calls"] += 1
            stats[f"{prefix}denoiser_batch_elements"] += n
        return torch.cat(outputs)

    def euler(states, velocity, i):
        dt = float(sigmas[i + 1]) - float(sigmas[i])
        return (states.float() + dt * velocity.float()).to(velocity.dtype)

    def rollout(states, after_step):
        for i in range(after_step, config.steps):
            velocity = predict(states, timesteps[i], rollout=True)
            states = euler(states, velocity, i)
        return states

    def score(images, guidance=False):
        values = torch.as_tensor(reward(images, prompt), device=device, dtype=torch.float32)
        if values.shape != (len(images),) or not torch.isfinite(values).all():
            raise ValueError("Reward must return one finite scalar per image")
        stats["reward_calls"] += 1
        stats["reward_image_evaluations"] += len(images)
        if guidance:
            stats["guidance_reward_image_evaluations"] += len(images)
        return values

    try:
        for i, t in enumerate(timesteps):
            step_num = i + 1
            s, sn = float(sigmas[i]), float(sigmas[i + 1])
            use_probes = (config.method != "unguided" and step_num in rhos and config.rho > 0 and
                          (config.noise_level > 0 if config.twin_sampler == "flow-grpo" else
                           config.exploration > 0 and sn > 0))
            pred = predict(anchors, t)
            stats["sequential_nfe"] += 1
            base = euler(anchors, pred, i)
            displacement = base - anchors
            trace = {"step": step_num, "sigma": s, "sigma_next": sn,
                     "guided": False, "guidance_eval": "final-rollout"}
            preview = None
            if use_probes:
                repeats = (config.probes,) + (1,) * (anchors.ndim - 1)
                states = anchors.repeat(repeats)
                velocities = pred.repeat(repeats)
                noise = torch.randn(states.shape, generator=probe_rng, device=device, dtype=torch.float32)
                if config.twin_sampler == "flow-grpo":
                    probes = flow_grpo_step(states, velocities, s, sn, config.noise_level,
                                            noise, sigma_cap=float(sigmas[1]))
                else:
                    probes = stochastic_flow_step(states, velocities, s, sn, config.exploration, noise)
                probes = probes.to(base.dtype)
                completed = rollout(torch.cat([base, probes]), step_num)
                preview = decode_images(pipe, completed, config.decode_batch_size)
                scores = score(preview, guidance=True)
                pair_dirs = fd_direction(completed[:k].repeat(repeats), completed[k:],
                                         scores[:k].repeat(config.probes), scores[k:])
                direction = pair_dirs.reshape(config.probes, k, *anchors.shape[1:]).mean(0)
                field = stein_field(base, direction, config.repulsion) if config.method == "fd-sgs" else direction
                correction = relative_correction(field, displacement, rhos[step_num], config.trust_ratio)
                base = base + correction.to(base.dtype)
                trace.update(guided=True, anchor_rewards=scores[:k].tolist(), twin_rewards=scores[k:].tolist(),
                             guidance_strength=rhos[step_num],
                             correction_ratios=(rms(correction) / (rms(displacement) + 1e-8)).flatten().tolist())
            anchors = base
            if not torch.isfinite(anchors).all():
                raise FloatingPointError(f"Nonfinite latent at step {step_num}")
            stats["steps"].append(trace)
            if on_step is not None:
                on_step(dict(trace), preview)
        images = decode_images(pipe, anchors, config.decode_batch_size)
        rewards = score(images)
        if cuda:
            torch.cuda.synchronize(device)
        stats.update(wall_seconds=time.perf_counter() - start,
                     total_denoiser_forward_calls=stats["denoiser_forward_calls"] + stats["rollout_denoiser_forward_calls"],
                     total_denoiser_batch_elements=stats["denoiser_batch_elements"] + stats["rollout_denoiser_batch_elements"],
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
                     final_rewards=rewards.tolist(), best_particle=int(rewards.argmax()))
        return images, stats
    finally:
        pipe.maybe_free_model_hooks()

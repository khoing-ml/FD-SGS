"""Eight effective Euler steps with persistent stochastic probes per anchor."""

from dataclasses import dataclass
import math
import time

import torch

from .core import fd_direction, flow_grpo_step, relative_correction, rms, stein_field, stochastic_flow_step


@dataclass(frozen=True)
class SamplingConfig:
    particles: int = 4
    steps: int = 8
    guidance_steps: tuple[int, ...] = (1, 2, 3, 4, 5, 6)  # one-based, effective model steps
    rho: float = 0.025
    repulsion: float = 0.1
    exploration: float = 0.0025
    twin_sampler: str = "fdfo"
    noise_level: float = 0.05
    probes: int = 1
    trust_ratio: float = 0.1
    method: str = "fd-sgs"
    seed: int = 42
    height: int = 1024
    width: int = 1024
    decode_batch_size: int = 1

    def __post_init__(self):
        if self.method not in {"fd-sgs", "independent-fd", "unguided"}:
            raise ValueError("Unknown sampling method")
        if self.twin_sampler not in {"fdfo", "edm", "flow-grpo"}:
            raise ValueError("twin_sampler must be fdfo or flow-grpo (edm is an alias for fdfo)")
        if not isinstance(self.probes, int) or self.probes < 1:
            raise ValueError("probes must be a positive integer")
        if self.steps != 8:
            raise ValueError("This initial Z-Image-Turbo experiment supports exactly 8 effective steps")
        if self.particles < 1 or self.decode_batch_size < 1:
            raise ValueError("Particle and decode batch sizes must be positive")
        if self.height < 16 or self.width < 16 or self.height % 16 or self.width % 16:
            raise ValueError("Height and width must be positive multiples of 16")
        if len(set(self.guidance_steps)) != len(self.guidance_steps) or any(
            n < 1 or n > self.steps for n in self.guidance_steps
        ):
            raise ValueError("Guidance steps must be unique one-based indices in 1..8")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.rho, self.repulsion, self.exploration, self.trust_ratio, self.noise_level)):
            raise ValueError("Guidance and exploration strengths must be finite and nonnegative")


def decode_images(pipe, latents, batch_size):
    images = []
    for batch in latents.split(batch_size):
        z = batch.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        z = z + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(z, return_dict=False)[0]
        images.extend(pipe.image_processor.postprocess(decoded, output_type="pil"))
    return images


@torch.inference_mode()
def sample(pipe, prompt, reward, config=SamplingConfig(), *, on_step=None):
    """Return PIL anchors, final rewards and auditable compute/step diagnostics.

    reward(images, prompt) must return one finite scalar per PIL image; higher
    is better. It may be nondifferentiable. No backpropagation is performed.
    The pipeline must use the original deterministic FlowMatch Euler scheduler.
    Optional on_step(trace, clean_images) receives diagnostics after each step;
    clean_images contains existing reward previews at guided steps, otherwise None.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift

    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError("Requires FlowMatchEulerDiscreteScheduler")
    if pipe.scheduler.config.get("stochastic_sampling", False) or pipe.scheduler.config.get("invert_sigmas", False):
        raise ValueError("Requires deterministic, descending-sigma Euler scheduling")
    device = torch.device(pipe._execution_device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    # Separate streams keep anchor initialization identical across all methods.
    probe_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, device=device)
    k = config.particles
    anchors = pipe.prepare_latents(k, pipe.transformer.in_channels, config.height,
                                  config.width, torch.float32, device, generator, None)
    paired = config.method != "unguided"
    # Probe-major layout: all K anchors, then K states for each probe index.
    repeats = (config.probes,) + (1,) * (anchors.ndim - 1)
    twins = anchors.repeat(repeats) if paired else None
    seq_len = (anchors.shape[2] // 2) * (anchors.shape[3] // 2)
    sc = pipe.scheduler.config
    mu = calculate_shift(seq_len, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                         sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
    pipe.scheduler.sigma_min = 0.0
    # The native 9-step schedule has 8 nonzero intervals and one zero interval.
    pipe.scheduler.set_timesteps(config.steps + 1, device=device, mu=mu)
    sigmas = pipe.scheduler.sigmas
    intervals = [(i, t) for i, t in enumerate(pipe.scheduler.timesteps)
                 if float(sigmas[i]) > float(sigmas[i + 1])]
    if len(intervals) != config.steps or float(sigmas[-1]) != 0:
        raise ValueError("Scheduler must produce exactly eight nonzero intervals ending at zero")
    stats = {"sequential_nfe": 0, "denoiser_forward_calls": 0,
             "denoiser_batch_elements": 0, "particles": k,
             "twin_trajectories": k * config.probes if paired else 0,
             "probes_per_particle": config.probes if paired else 0,
             "twin_sampler": config.twin_sampler if paired else None,
             "reward_calls": 0, "reward_image_evaluations": 0,
             "guidance_reward_image_evaluations": 0, "steps": []}

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
        for step, (i, t) in enumerate(intervals, 1):
            states = torch.cat([anchors, twins]) if paired else anchors
            inputs = list(states.to(pipe.transformer.dtype).unsqueeze(2).unbind(0))
            # Z-Image predicts velocity in increasing data-time, scheduler uses noise-time.
            model_time = (1000 - t.expand(len(states))) / 1000
            output = pipe.transformer(inputs, model_time, embeds * len(states), return_dict=False)[0]
            velocity = -torch.stack([v.float() for v in output]).squeeze(2)
            stats["sequential_nfe"] += 1
            stats["denoiser_forward_calls"] += 1
            stats["denoiser_batch_elements"] += len(states)
            s, sn = float(sigmas[i]), float(sigmas[i + 1])
            base = pipe.scheduler.step(velocity[:k], t, anchors, return_dict=False)[0]
            displacement = base - anchors
            trace = {"step": step, "sigma": s, "sigma_next": sn, "guided": False}
            clean_images = None
            correction = torch.zeros_like(base)
            if paired:
                noise = torch.randn(twins.shape, generator=probe_generator, device=device, dtype=torch.float32)
                if config.twin_sampler == "flow-grpo":
                    probe_base = flow_grpo_step(twins, velocity[k:], s, sn, config.noise_level,
                                                noise, sigma_cap=float(sigmas[1]))
                else:  # FDFO overshoot/re-noise; "edm" is a legacy alias.
                    probe_base = stochastic_flow_step(twins, velocity[k:], s, sn, config.exploration, noise)
            if paired and step in config.guidance_steps and config.rho > 0:
                # Score the probe after this step's stochastic transition so the
                # first step has a finite-difference signal despite shared initial noise.
                # The anchor endpoint prediction equals anchors - s * velocity[:k].
                clean = torch.cat([base - sn * velocity[:k],
                                   probe_base - sn * velocity[k:]])
                clean_images = decode_images(pipe, clean, config.decode_batch_size)
                scores = score(clean_images, guidance=True)
                pair_directions = fd_direction(clean[:k].repeat(repeats), clean[k:],
                                               scores[:k].repeat(config.probes), scores[k:])
                direction = pair_directions.reshape(config.probes, k, *anchors.shape[1:]).mean(0)
                field = (stein_field(base, direction, config.repulsion)
                         if config.method == "fd-sgs" else direction)
                correction = relative_correction(field, displacement, config.rho, config.trust_ratio)
                trace.update(guided=True, anchor_rewards=scores[:k].tolist(),
                             twin_rewards=scores[k:].tolist(),
                             correction_ratios=(rms(correction) / (rms(displacement) + 1e-8)).flatten().tolist())
            if paired:
                # Translate each anchor's probes by the SAME correction so guidance itself
                # does not become an uncontrolled source of pair separation.
                twins = probe_base + correction.repeat(repeats)
            anchors = base + correction
            if not torch.isfinite(anchors).all() or (paired and not torch.isfinite(twins).all()):
                raise FloatingPointError(f"Nonfinite latent at step {step}")
            stats["steps"].append(trace)
            if on_step is not None:
                on_step(dict(trace), clean_images)
        images = decode_images(pipe, anchors, config.decode_batch_size)
        scores = score(images)
        if cuda:
            torch.cuda.synchronize(device)
        stats.update(wall_seconds=time.perf_counter() - start,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
                     final_rewards=scores.tolist(), best_particle=int(scores.argmax()))
        return images, stats
    finally:
        pipe.maybe_free_model_hooks()

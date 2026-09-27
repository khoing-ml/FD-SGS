"""Hyper-SD CFG-LoRA sampling with DDIM variance probes.

The probe uses the scheduler's discrete diffusion variance. It deliberately
does not reuse the flow-matching FDFO/Flow-GRPO transition equations.
"""

from dataclasses import dataclass
import math
import time

import torch
from diffusers import DDIMScheduler

from .core import fd_direction, relative_correction, rms, stein_field


HYPER_MODELS = {
    "hyper-sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "hyper-sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "hyper-sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
}
HYPER_LORAS = {
    ("hyper-sdxl", 8): "Hyper-SDXL-8steps-CFG-lora.safetensors",
    ("hyper-sdxl", 12): "Hyper-SDXL-12steps-CFG-lora.safetensors",
    ("hyper-sd15", 8): "Hyper-SD15-8steps-CFG-lora.safetensors",
    ("hyper-sd15", 12): "Hyper-SD15-12steps-CFG-lora.safetensors",
    ("hyper-sd3", 4): "Hyper-SD3-4steps-CFG-lora.safetensors",
    ("hyper-sd3", 8): "Hyper-SD3-8steps-CFG-lora.safetensors",
    ("hyper-sd3", 16): "Hyper-SD3-16steps-CFG-lora.safetensors",
}
LORA_REPO = "ByteDance/Hyper-SD"


@dataclass(frozen=True)
class HyperSamplingConfig:
    particles: int = 4
    steps: int = 8
    guidance_steps: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    guidance_eval: str = "final-rollout"
    rho: float = 0.025
    rho_schedule: str = "linear-decay"
    rho_start_multiplier: float = 2.0
    repulsion: float = 0.1
    probes: int = 1
    trust_ratio: float = 0.1
    ddim_eta: float = 0.5
    cfg_scale: float = 5.0
    method: str = "fd-sgs"
    seed: int = 42
    height: int = 1024
    width: int = 1024
    decode_batch_size: int = 1
    unet_batch_size: int = 1
    backbone: str = "hyper-sdxl"

    def __post_init__(self):
        if self.backbone not in {"hyper-sdxl", "hyper-sd15"}:
            raise ValueError("DDIM sampler supports Hyper-SDXL or Hyper-SD1.5")
        if (self.backbone, self.steps) not in HYPER_LORAS:
            raise ValueError("SDXL and SD1.5 CFG LoRAs support 8 or 12 steps")
        if self.method not in {"fd-sgs", "independent-fd", "unguided"}:
            raise ValueError("Unknown sampling method")
        if self.guidance_eval != "final-rollout":
            raise ValueError("Hyper-SD guidance requires final-rollout reward evaluation")
        if self.rho_schedule not in {"linear-decay", "constant"}:
            raise ValueError("Unknown rho schedule")
        if any(not isinstance(value, int) or value < 1 for value in
               (self.particles, self.probes, self.decode_batch_size, self.unet_batch_size)):
            raise ValueError("Particles, probes and batch sizes must be positive integers")
        if self.height < 16 or self.width < 16 or self.height % 8 or self.width % 8:
            raise ValueError("Height and width must be positive multiples of 8")
        if len(set(self.guidance_steps)) != len(self.guidance_steps) or any(s < 1 or s > self.steps for s in self.guidance_steps):
            raise ValueError(f"Guidance steps must be unique one-based indices in 1..{self.steps}")
        if self.steps in self.guidance_steps:
            raise ValueError("DDIM's final step has zero probe variance; guide before the final step")
        if any(not math.isfinite(v) or v < 0 for v in (self.rho, self.repulsion, self.trust_ratio, self.ddim_eta)) or self.ddim_eta > 1:
            raise ValueError("Guidance and DDIM eta must be finite and nonnegative")
        if not math.isfinite(self.rho_start_multiplier) or self.rho_start_multiplier < 1:
            raise ValueError("rho_start_multiplier must be finite and at least 1")
        if not 0 < self.cfg_scale or not math.isfinite(self.cfg_scale):
            raise ValueError("CFG scale must be finite and positive")


def decode_images(pipe, latents, batch_size):
    images = []
    for batch in latents.split(batch_size):
        vae = pipe.vae
        upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False) and hasattr(pipe, "upcast_vae")
        if upcast:
            pipe.upcast_vae()
            decode_dtype = next(iter(vae.post_quant_conv.parameters())).dtype
        else:
            decode_dtype = vae.dtype
        z = batch.to(decode_dtype) / vae.config.scaling_factor
        if getattr(vae.config, "latents_mean", None) is not None and getattr(vae.config, "latents_std", None) is not None:
            mean = torch.as_tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
            z = batch.to(decode_dtype) * std / vae.config.scaling_factor + mean
        decoded = vae.decode(z, return_dict=False)[0]
        images.extend(pipe.image_processor.postprocess(decoded, output_type="pil"))
        if upcast:
            vae.to(dtype=torch.float16)
    return images


@torch.inference_mode()
def sample(pipe, prompt, reward, config=HyperSamplingConfig(), *, on_step=None):
    if not isinstance(pipe.scheduler, DDIMScheduler):
        raise ValueError("Hyper-SD sampler requires DDIMScheduler")
    if pipe.scheduler.config.timestep_spacing != "trailing":
        raise ValueError("Hyper-SD requires trailing DDIM timesteps")
    device = torch.device(pipe._execution_device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    rng = torch.Generator(device=device).manual_seed(config.seed)
    probe_rng = torch.Generator(device=device).manual_seed(config.seed + 1)
    k = config.particles
    pipe.scheduler.set_timesteps(config.steps, device=device)
    timesteps = list(pipe.scheduler.timesteps)
    if len(timesteps) != config.steps:
        raise ValueError(f"DDIM scheduler did not produce {config.steps} timesteps")
    sdxl = config.backbone == "hyper-sdxl"
    if sdxl:
        pos, neg, pooled_pos, pooled_neg = pipe.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
        projection_dim = (pipe.text_encoder_2.config.projection_dim if pipe.text_encoder_2 is not None
                          else pooled_pos.shape[-1])
        time_ids = pipe._get_add_time_ids((config.height, config.width), (0, 0),
                                           (config.height, config.width), dtype=pos.dtype,
                                           text_encoder_projection_dim=projection_dim).to(device)
    else:
        pos, neg = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                      do_classifier_free_guidance=True, negative_prompt=None)
    anchors = pipe.prepare_latents(k, pipe.unet.config.in_channels, config.height, config.width,
                                   pos.dtype, device, rng, None)
    guided = sorted(config.guidance_steps)
    rhos = {step: config.rho if config.rho_schedule == "constant" else
            config.rho * (1 + (config.rho_start_multiplier - 1) *
                          (1 - rank / max(len(guided) - 1, 1)))
            for rank, step in enumerate(guided)}
    stats = {"sequential_nfe": 0, "denoiser_forward_calls": 0, "denoiser_batch_elements": 0,
             "rollout_denoiser_forward_calls": 0, "rollout_denoiser_batch_elements": 0,
             "particles": k, "twin_trajectories": k * config.probes if config.method != "unguided" else 0,
             "probes_per_particle": config.probes if config.method != "unguided" else 0,
             "twin_sampler": "ddim" if config.method != "unguided" else None,
             "reward_calls": 0, "reward_image_evaluations": 0,
             "guidance_reward_image_evaluations": 0, "steps": []}

    def predict(states, t, rollout=False):
        outputs = []
        for chunk in states.split(config.unet_batch_size):
            n = len(chunk)
            model_input = pipe.scheduler.scale_model_input(torch.cat([chunk, chunk]), t)
            embeds = torch.cat([neg.expand(n, -1, -1), pos.expand(n, -1, -1)])
            added = None
            if sdxl:
                added = {"text_embeds": torch.cat([pooled_neg.expand(n, -1), pooled_pos.expand(n, -1)]),
                         "time_ids": time_ids.expand(2 * n, -1)}
            pred = pipe.unet(model_input, t, encoder_hidden_states=embeds,
                             added_cond_kwargs=added, return_dict=False)[0]
            uncond, cond = pred.chunk(2)
            outputs.append(uncond + config.cfg_scale * (cond - uncond))
            prefix = "rollout_" if rollout else ""
            stats[f"{prefix}denoiser_forward_calls"] += 1
            stats[f"{prefix}denoiser_batch_elements"] += n
        return torch.cat(outputs)

    def step(states, t, eta=0, noise=None, prediction=None, rollout=False):
        if prediction is None:
            prediction = predict(states, t, rollout=rollout)
        return pipe.scheduler.step(prediction, t, states, eta=eta, variance_noise=noise,
                                   return_dict=False)[0]

    def rollout(states, after_step):
        for t in timesteps[after_step:]:
            states = step(states, t, rollout=True)
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
        for index, t in enumerate(timesteps):
            step_num = index + 1
            use_probes = config.method != "unguided" and step_num in rhos and config.rho > 0 and config.ddim_eta > 0
            repeats = (config.probes,) + (1,) * (anchors.ndim - 1)
            pred = predict(anchors, t)
            stats["sequential_nfe"] += 1
            base = step(anchors, t, prediction=pred[:k])
            displacement = base - anchors
            current_t = int(t)
            next_t = int(timesteps[index + 1]) if index + 1 < len(timesteps) else None
            trace = {"step": step_num, "sigma": float((1 - pipe.scheduler.alphas_cumprod[current_t]).sqrt()),
                     "sigma_next": float((1 - pipe.scheduler.alphas_cumprod[next_t]).sqrt())
                     if index + 1 < len(timesteps) else 0.0,
                     "guided": False, "guidance_eval": "final-rollout"}
            preview = None
            if use_probes:
                probe_noise = torch.randn((k * config.probes, *anchors.shape[1:]), generator=probe_rng,
                                          device=device, dtype=anchors.dtype)
                probes = step(anchors.repeat(repeats), t, eta=config.ddim_eta,
                              noise=probe_noise, prediction=pred.repeat(repeats))
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

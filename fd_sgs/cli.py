import argparse
from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import time


def main():
    parser = argparse.ArgumentParser(description="FD-SGS: Z-Image-Turbo and Hyper-SD CFG LoRAs")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-dataset", choices=["image-reward", "hpsv2"])
    prompts.add_argument("--prompts-file", type=Path, help="TXT lines or JSON list of {id, prompt} records")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print selected prompts/settings without loading models or starting W&B")
    parser.add_argument("--backbone", choices=["zimage", "hyper-sdxl", "hyper-sd15", "hyper-sd3"], default="zimage")
    parser.add_argument("--steps", type=int, help="Checkpoint-matched steps: Z-Image 8; SDXL/SD1.5 8 or 12; SD3 4, 8, or 16")
    parser.add_argument("--model", help="Base model override; defaults to the selected backbone's official base")
    parser.add_argument("--hyper-lora-repo", default="ByteDance/Hyper-SD")
    parser.add_argument("--lora-scale", type=float, help="Hyper-SD fusion scale: default 1 for SDXL/SD1.5, 0.125 for SD3")
    parser.add_argument("--ddim-eta", type=float, default=0.5, help="Hyper-SD probe variance (0=deterministic, 1=DDPM-like)")
    parser.add_argument("--cfg-scale", type=float, help="Hyper-SD CFG scale: default 5 for SDXL/SD1.5; 3/5/7 for SD3 4/8/16 steps")
    parser.add_argument("--method", choices=["fd-sgs", "independent-fd", "unguided", "compare"], default="fd-sgs")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1, help="Stochastic probes per anchor; guidance lookahead processes K*(1+B) states")
    parser.add_argument("--twin-sampler", choices=["fdfo", "flow-grpo", "edm", "ddim"],
                        help="Probe transition: default FDFO for Z-Image, DDIM for SDXL/SD1.5, Flow-GRPO for SD3")
    parser.add_argument("--noise-level", type=float, default=0.05, help="Flow-GRPO diffusion strength for Z-Image/SD3; 0 removes probe noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--guidance-steps", type=int, nargs="+", help="One-based steps for reward guidance; default first six or fewer")
    parser.add_argument("--guidance-eval", choices=["predicted-clean", "final-rollout"],
                        help="Reward intermediate clean estimate or completed deterministic lookahead; Hyper-SD requires final-rollout")
    parser.add_argument("--rho", type=float, default=0.025, help="Final guided-step strength; also all steps with constant schedule")
    parser.add_argument("--rho-schedule", choices=["linear-decay", "constant"], default="linear-decay")
    parser.add_argument("--rho-start-multiplier", type=float, default=2.0,
                        help="Initial guided-step strength as a multiple of --rho (linear-decay only)")
    parser.add_argument("--repulsion", type=float, default=0.1)
    parser.add_argument("--exploration", type=float, default=0.0025, help="FDFO exploration gamma (only for --twin-sampler fdfo/edm)")
    parser.add_argument("--trust-ratio", type=float, default=0.1)
    parser.add_argument("--reward", default="pickscore", help="Guidance reward: imagereward, pickscore, hpsv2, clipscore, brightness, or module:function")
    parser.add_argument("--eval-rewards", nargs="+", default=[], help="Additional final-image scorers, or all for the four benchmark rewards")
    parser.add_argument("--hps-version", choices=["v2.0", "v2.1"], default="v2.0")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--unet-batch-size", type=int, default=1,
                        help="Hyper-SD: states per UNet or transformer call (CFG doubles the model batch)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--offload", action="store_true", help="Offload pipeline components to CPU between calls")
    parser.add_argument("--output", type=Path, default=Path("outputs/zimage"))
    parser.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    parser.add_argument("--wandb-project", default="fd-sgs")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-image-limit", type=int, default=8, help="Maximum gallery images (winner always included)")
    parser.add_argument("--wandb-previews", action="store_true", help="Log scored anchor/probe images at guided steps")
    args = parser.parse_args()

    import torch
    import diffusers
    import transformers
    from .sampler import SamplingConfig
    from .hyper_sd import HYPER_MODELS, HYPER_LORAS, HyperSamplingConfig
    from .hyper_sd3 import HyperSD3SamplingConfig
    from .tracking import WandbLogger
    from .prompts import load_prompts
    from .rewards import REWARD_NAMES, RewardRegistry, normalize_reward

    try:
        if args.backbone == "zimage":
            args.steps = 8 if args.steps is None else args.steps
            if args.steps != 8:
                parser.error("Z-Image-Turbo supports exactly 8 effective steps")
            args.model = args.model or "Tongyi-MAI/Z-Image-Turbo"
            args.dtype = args.dtype or "bfloat16"
            args.twin_sampler = args.twin_sampler or "fdfo"
            args.guidance_eval = args.guidance_eval or "predicted-clean"
            args.height = 1024 if args.height is None else args.height
            args.width = 1024 if args.width is None else args.width
            if args.twin_sampler == "ddim":
                parser.error("--twin-sampler ddim requires a Hyper-SD backbone")
        else:
            args.steps = (16 if args.backbone == "hyper-sd3" else 8) if args.steps is None else args.steps
            if (args.backbone, args.steps) not in HYPER_LORAS:
                valid = sorted(s for backbone, s in HYPER_LORAS if backbone == args.backbone)
                parser.error(f"{args.backbone} supports --steps {'/'.join(map(str, valid))}")
            args.model = args.model or HYPER_MODELS[args.backbone]
            args.dtype = args.dtype or "float16"
            args.twin_sampler = args.twin_sampler or ("flow-grpo" if args.backbone == "hyper-sd3" else "ddim")
            args.guidance_eval = args.guidance_eval or "final-rollout"
            size = 512 if args.backbone == "hyper-sd15" else 1024
            args.height = size if args.height is None else args.height
            args.width = size if args.width is None else args.width
            args.lora_scale = args.lora_scale if args.lora_scale is not None else (0.125 if args.backbone == "hyper-sd3" else 1.0)
            args.cfg_scale = args.cfg_scale if args.cfg_scale is not None else (
                {4: 3.0, 8: 5.0, 16: 7.0}[args.steps] if args.backbone == "hyper-sd3" else 5.0)
            if not math.isfinite(args.lora_scale) or args.lora_scale <= 0:
                parser.error("--lora-scale must be finite and positive")
            if args.backbone == "hyper-sd3":
                if args.twin_sampler not in {"flow-grpo", "fdfo"}:
                    parser.error("Hyper-SD3 requires --twin-sampler flow-grpo or fdfo")
            elif args.twin_sampler != "ddim":
                parser.error("Hyper-SDXL/SD1.5 require --twin-sampler ddim")
            if args.guidance_eval != "final-rollout":
                parser.error("Hyper-SD requires --guidance-eval final-rollout")
            if args.backbone != "hyper-sd3" and not 0 <= args.ddim_eta <= 1:
                parser.error("--ddim-eta must be in [0, 1]")
            if args.backbone != "hyper-sd3" and not 5 <= args.cfg_scale <= 8:
                parser.error("--cfg-scale must be in the SDXL/SD1.5 CFG-LoRA recommended range [5, 8]")
            if args.backbone == "hyper-sd3" and (not math.isfinite(args.cfg_scale) or args.cfg_scale <= 1):
                parser.error("--cfg-scale must be finite and greater than 1 for SD3")
        args.guidance_steps = args.guidance_steps or list(range(1, min(args.steps, 6) + 1))
        records, source = load_prompts(prompt=args.prompt, dataset=args.prompt_dataset, path=args.prompts_file,
                                       start=args.start_index, limit=args.max_prompts)
        args.reward = normalize_reward(args.reward)
        args.eval_rewards = list(dict.fromkeys(name for value in args.eval_rewards
                                 for name in (REWARD_NAMES if value == "all" else [normalize_reward(value)])))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    methods = ["unguided", "independent-fd", "fd-sgs"] if args.method == "compare" else [args.method]
    config_type = (SamplingConfig if args.backbone == "zimage" else
                   HyperSD3SamplingConfig if args.backbone == "hyper-sd3" else HyperSamplingConfig)
    common = dict(particles=args.particles, steps=args.steps, seed=args.seed, height=args.height,
                  width=args.width, guidance_steps=tuple(args.guidance_steps),
                  guidance_eval=args.guidance_eval, rho=args.rho,
                  rho_schedule=args.rho_schedule,
                  rho_start_multiplier=args.rho_start_multiplier,
                  repulsion=args.repulsion, probes=args.probes,
                  trust_ratio=args.trust_ratio, decode_batch_size=args.decode_batch_size)
    if args.backbone == "zimage":
        specific = dict(exploration=args.exploration, twin_sampler=args.twin_sampler, noise_level=args.noise_level)
    elif args.backbone == "hyper-sd3":
        specific = dict(cfg_scale=args.cfg_scale, backbone=args.backbone,
                        twin_sampler=args.twin_sampler, noise_level=args.noise_level,
                        exploration=args.exploration, unet_batch_size=args.unet_batch_size)
    else:
        specific = dict(ddim_eta=args.ddim_eta, cfg_scale=args.cfg_scale,
                        backbone=args.backbone, unet_batch_size=args.unet_batch_size)
    try:
        configs = [config_type(method=m, **common, **specific) for m in methods]
    except ValueError as exc:
        parser.error(str(exc))
    if args.reward_batch_size < 1:
        parser.error("--reward-batch-size must be positive")
    if args.wandb_image_limit < 1:
        parser.error("--wandb-image-limit must be positive")
    if args.offload and torch.device(args.device).type != "cuda":
        parser.error("--offload requires a CUDA device")
    # Refuse accidental overwrite before downloading/loading large models.
    batch = args.prompt is None
    if not args.dry_run:
        for record in records:
            output = args.output / f"prompt_{record.index:05d}" if batch else args.output
            for method in methods:
                target = output / method
                if target.exists() and any(target.iterdir()):
                    parser.error(f"Output directory is not empty: {target}; choose a new --output")
        if batch and (args.output / "results.csv").exists():
            parser.error("Batch results already exist; choose a new --output")
    registry = RewardRegistry(args.reward_device, args.reward_batch_size, args.hps_version)
    scorer_names = list(dict.fromkeys([args.reward, *args.eval_rewards]))
    if args.dry_run:
        print(json.dumps({"source": source, "prompts": [asdict(r) for r in records],
                          "seed_rule": "seed + prompt index", "base_seed": args.seed,
                          "configs": [asdict(c) for c in configs], "scorers": registry.describe(scorer_names)}, indent=2))
        return
    # Check optional backend installation before downloading the generation model.
    registry.check_available(scorer_names)
    pipe, rows = None, []
    for record in records:
        run_args = argparse.Namespace(**vars(args))
        run_args.prompt = record.text
        run_args.output = args.output / f"prompt_{record.index:05d}" if batch else args.output
        run_args.seed = args.seed + record.index
        run_configs = [replace(c, seed=run_args.seed) for c in configs]
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(run_args).items()}
        run_config.update(prompt_id=record.id, prompt_index=record.index, prompt_source=source,
                          scorers=registry.describe(scorer_names), sampling_configs=[asdict(c) for c in run_configs],
                          torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__)
        name = f"{args.wandb_name or args.output.name}-{record.index:05d}" if batch else args.wandb_name
        group = args.wandb_group or (args.output.name if batch else None)
        with WandbLogger(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
                         name=name, group=group, tags=args.wandb_tags, directory=run_args.output,
                         config=run_config, image_limit=args.wandb_image_limit, previews=args.wandb_previews) as tracker:
            if pipe is None:
                pipe = _load_pipeline(args)
            summary = _execute(run_args, run_configs, tracker, pipe, registry, record, source)
        if batch:
            for method, stats in summary.items():
                for particle, score in enumerate(stats["final_rewards"]):
                    rows.append({"prompt_index": record.index, "prompt_id": record.id, "prompt": record.text,
                                 "seed": run_args.seed, "method": method, "particle": particle,
                                 "selected_by_guidance_reward": particle == stats["best_particle"],
                                 "guidance_reward": score,
                                 **{name: scores[particle] for name, scores in stats["evaluation_scores"].items()}})
            _write_batch_results(args.output, rows, scorer_names, source)


def _load_pipeline(args):
    import torch
    if args.backbone == "zimage":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    else:
        import importlib.util
        if importlib.util.find_spec("peft") is None:
            raise RuntimeError("Hyper-SD LoRA loading requires PEFT; install with pip install -e '.[hyper]'")
        from diffusers import (DDIMScheduler, FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline,
                               StableDiffusionXLPipeline, StableDiffusion3Pipeline)
        from .hyper_sd import HYPER_MODELS, HYPER_LORAS
        pipeline_type = (StableDiffusion3Pipeline if args.backbone == "hyper-sd3" else
                         StableDiffusionXLPipeline if args.backbone == "hyper-sdxl" else StableDiffusionPipeline)
        load_kwargs = {"torch_dtype": getattr(torch, args.dtype)}
        if args.dtype == "float16" and args.model == HYPER_MODELS[args.backbone] and args.backbone != "hyper-sd3":
            load_kwargs["variant"] = "fp16"
        pipe = pipeline_type.from_pretrained(args.model, **load_kwargs)
        pipe.load_lora_weights(args.hyper_lora_repo, weight_name=HYPER_LORAS[(args.backbone, args.steps)])
        pipe.fuse_lora(lora_scale=args.lora_scale)
        if args.backbone == "hyper-sd3":
            if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
                raise ValueError("Hyper-SD3 base model must use FlowMatchEulerDiscreteScheduler")
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if args.offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    return pipe


def _execute(args, configs, tracker, pipe, registry, record, source):
    import torch
    import diffusers
    import transformers
    if args.backbone == "zimage":
        from .sampler import sample
    elif args.backbone == "hyper-sd3":
        from .hyper_sd3 import sample
    else:
        from .hyper_sd import sample

    reward = registry.get(args.reward)
    summary = {}
    for config in configs:
        probe_info = (f", {config.probes} {args.twin_sampler} probes per anchor"
                      if config.method != "unguided" else "")
        print(f"Running {config.method}: {config.particles} particles{probe_info}, {config.steps} steps", flush=True)
        images, stats = sample(pipe, args.prompt, reward, config,
                               on_step=tracker.step_callback(config.method))
        target = args.output / config.method
        target.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(target / f"particle_{i:02d}.png")
        images[stats["best_particle"]].save(target / "best.png")
        evaluation_start = time.perf_counter()
        evaluation_scores = {args.reward: stats["final_rewards"]}
        for name in args.eval_rewards:
            if name not in evaluation_scores:
                print(f"Evaluating {name} on {len(images)} final images", flush=True)
                evaluation_scores[name] = registry.score(name, images, args.prompt)
        stats.update(evaluation_scores=evaluation_scores,
                     evaluation_wall_seconds=time.perf_counter() - evaluation_start,
                     evaluation_reward_calls=len(evaluation_scores) - 1,
                     evaluation_reward_image_evaluations=(len(evaluation_scores) - 1) * len(images))
        metadata = {"model": args.model, "prompt": args.prompt, "reward": args.reward,
                    "prompt_id": record.id, "prompt_index": record.index, "prompt_source": source,
                    "scorers": registry.describe(evaluation_scores),
                    "reward_device": args.reward_device, "dtype": args.dtype, "offload": args.offload,
                    "torch": torch.__version__, "diffusers": diffusers.__version__,
                    "transformers": transformers.__version__, "config": asdict(config), "metrics": stats}
        (target / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary[config.method] = {key: value for key, value in stats.items() if key != "steps"}
        tracker.log_result(config.method, images, stats, target)
        print(f"Saved {target}; best reward={max(stats['final_rewards']):.5f}", flush=True)
    if args.method == "compare":
        (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        tracker.log_comparison()
    return summary


def _write_batch_results(output, rows, scorer_names, source):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        samples = [row for row in rows if row["method"] == method]
        selected = [row for row in samples if row["selected_by_guidance_reward"]]
        summary[method] = {"prompts": len(selected), "images": len(samples), "rewards": {
            name: {"all_particles_mean": fmean(row[name] for row in samples),
                   "selected_particles_mean": fmean(row[name] for row in selected)} for name in scorer_names}}
    (output / "batch_summary.json").write_text(json.dumps({"source": source, "methods": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()

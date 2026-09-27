import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
from statistics import fmean
import time


def main():
    parser = argparse.ArgumentParser(description="FD-SGS for Z-Image-Turbo: eight effective model steps")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-dataset", choices=["image-reward", "hpsv2"])
    prompts.add_argument("--prompts-file", type=Path, help="TXT lines or JSON list of {id, prompt} records")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print selected prompts/settings without loading models or starting W&B")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--method", choices=["fd-sgs", "independent-fd", "unguided", "compare"], default="fd-sgs")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1, help="Stochastic probes per anchor; guided batch size is K*(1+B)")
    parser.add_argument("--twin-sampler", choices=["fdfo", "flow-grpo", "edm"], default="fdfo",
                        help="Probe transition: FDFO overshoot/re-noise (default), Flow-GRPO SDE, or legacy edm alias")
    parser.add_argument("--noise-level", type=float, default=0.05, help="Flow-GRPO diffusion strength; 0 gives Euler ODE probes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-steps", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--rho", type=float, default=0.025)
    parser.add_argument("--repulsion", type=float, default=0.1)
    parser.add_argument("--exploration", type=float, default=0.0025, help="FDFO exploration gamma (only for --twin-sampler fdfo/edm)")
    parser.add_argument("--trust-ratio", type=float, default=0.1)
    parser.add_argument("--reward", default="pickscore", help="Guidance reward: imagereward, pickscore, hpsv2, clipscore, brightness, or module:function")
    parser.add_argument("--eval-rewards", nargs="+", default=[], help="Additional final-image scorers, or all for the four benchmark rewards")
    parser.add_argument("--hps-version", choices=["v2.0", "v2.1"], default="v2.0")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--offload", action="store_true", help="Offload pipeline components to CPU between calls")
    parser.add_argument("--output", type=Path, default=Path("outputs/zimage"))
    parser.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    parser.add_argument("--wandb-project", default="fd-sgs")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-image-limit", type=int, default=8, help="Maximum gallery images (winner always included)")
    parser.add_argument("--wandb-previews", action="store_true", help="Log predicted-clean anchor/probe images at guided steps")
    args = parser.parse_args()

    import torch
    import diffusers
    import transformers
    from .sampler import SamplingConfig
    from .tracking import WandbLogger
    from .prompts import load_prompts
    from .rewards import REWARD_NAMES, RewardRegistry, normalize_reward

    try:
        records, source = load_prompts(prompt=args.prompt, dataset=args.prompt_dataset, path=args.prompts_file,
                                       start=args.start_index, limit=args.max_prompts)
        args.reward = normalize_reward(args.reward)
        args.eval_rewards = list(dict.fromkeys(name for value in args.eval_rewards
                                 for name in (REWARD_NAMES if value == "all" else [normalize_reward(value)])))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    methods = ["unguided", "independent-fd", "fd-sgs"] if args.method == "compare" else [args.method]
    configs = [SamplingConfig(particles=args.particles, method=m, seed=args.seed, height=args.height,
                              width=args.width, guidance_steps=tuple(args.guidance_steps), rho=args.rho,
                              repulsion=args.repulsion, exploration=args.exploration,
                              twin_sampler=args.twin_sampler, noise_level=args.noise_level, probes=args.probes,
                              trust_ratio=args.trust_ratio, decode_batch_size=args.decode_batch_size) for m in methods]
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
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
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
    from .sampler import sample

    reward = registry.get(args.reward)
    summary = {}
    for config in configs:
        probe_info = (f", {config.probes} {config.twin_sampler} probes per anchor"
                      if config.method != "unguided" else "")
        print(f"Running {config.method}: {config.particles} particles{probe_info}, 8 effective steps", flush=True)
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

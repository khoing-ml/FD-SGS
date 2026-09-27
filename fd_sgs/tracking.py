"""Optional W&B experiment logging; importing this module never starts a run."""

from pathlib import Path
from statistics import fmean, pstdev


class WandbLogger:
    def __init__(self, *, mode, project, config, directory, entity=None, name=None,
                 group=None, tags=None, image_limit=8, previews=False):
        self.run = None
        self.image_limit = image_limit
        self.previews = previews
        self.completed = {}
        self.evaluations = {}
        if mode == "disabled":
            return
        import wandb
        self.wandb = wandb
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=project, entity=entity, name=name, group=group,
                              tags=tags, mode=mode, config=config, dir=str(directory),
                              job_type="sampling")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            self.run.finish(exit_code=0 if exc_type is None else 1)
        return False

    def step_callback(self, method):
        if self.run is None:
            return None
        # Each method has its own x-axis, so compare mode can restart at step 1
        # without sending decreasing values to W&B's global history step.
        axis = f"{method}/step"
        self.run.define_metric(axis)
        self.run.define_metric(f"{method}/sampling/*", step_metric=axis)

        def log_step(trace, images):
            prefix = f"{method}/sampling"
            data = {axis: trace["step"], f"{prefix}/sigma": trace["sigma"],
                    f"{prefix}/sigma_next": trace["sigma_next"],
                    f"{prefix}/guided": int(trace["guided"])}
            if trace["guided"]:
                if "guidance_strength" in trace:
                    data[f"{prefix}/guidance_strength"] = trace["guidance_strength"]
                anchors, twins = trace["anchor_rewards"], trace["twin_rewards"]
                delta = [r - anchors[j % len(anchors)] for j, r in enumerate(twins)]
                for name, values in (("anchor_reward", anchors), ("probe_reward", twins),
                                     ("reward_difference", delta),
                                     ("correction_ratio", trace["correction_ratios"])):
                    data.update({f"{prefix}/{name}_mean": fmean(values),
                                 f"{prefix}/{name}_std": pstdev(values),
                                 f"{prefix}/{name}_min": min(values),
                                 f"{prefix}/{name}_max": max(values)})
                for i, score in enumerate(anchors):
                    data[f"{prefix}/anchor_{i:02d}_reward"] = score
                if self.previews and images is not None:
                    # Separate caps ensure both anchors and probes are visible.
                    k = len(anchors)
                    indices = list(range(min(k, self.image_limit)))
                    indices += list(range(k, min(len(images), k + self.image_limit)))
                    scores = anchors + twins
                    media_key = ("final_rollouts" if trace.get("guidance_eval") == "final-rollout"
                                 else "clean_predictions")
                    data[f"{prefix}/{media_key}"] = [self.wandb.Image(
                        images[j], caption=(f"step {trace['step']} | "
                        + (f"anchor {j}" if j < k else f"probe {(j-k)//k}, anchor {(j-k)%k}")
                        + f" | reward {scores[j]:.5f}")) for j in indices]
            self.run.log(data)
        return log_step

    def log_result(self, method, images, stats, directory):
        if self.run is None:
            return
        rewards = stats["final_rewards"]
        values = {"reward_mean": fmean(rewards), "reward_std": pstdev(rewards),
                  "reward_min": min(rewards), "reward_max": max(rewards),
                  "best_particle": stats["best_particle"]}
        for key in ("sequential_nfe", "denoiser_forward_calls", "denoiser_batch_elements",
                    "rollout_denoiser_forward_calls", "rollout_denoiser_batch_elements",
                    "total_denoiser_forward_calls", "total_denoiser_batch_elements",
                    "reward_calls", "reward_image_evaluations", "guidance_reward_image_evaluations",
                    "particles", "twin_trajectories", "probes_per_particle", "wall_seconds",
                    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            values[key] = stats[key]
        self.completed[method] = values
        self.run.summary.update({f"{method}/final/{key}": value for key, value in values.items()})
        best = stats["best_particle"]
        evaluation = stats.get("evaluation_scores", {})
        self.evaluations[method] = {}
        for name, scores in evaluation.items():
            metrics = {"mean": fmean(scores), "std": pstdev(scores), "min": min(scores),
                       "max": max(scores), "at_selected_particle": scores[best]}
            self.evaluations[method][name] = metrics
            self.run.summary.update({f"{method}/evaluation/{name}/{key}": value for key, value in metrics.items()})
        for key in ("evaluation_wall_seconds", "evaluation_reward_calls", "evaluation_reward_image_evaluations"):
            if key in stats:
                self.run.summary[f"{method}/final/{key}"] = stats[key]
        table = self.wandb.Table(columns=["particle", "reward", "best", *evaluation.keys(), "image"])
        # Always include the winner, even if it falls outside the image cap.
        shown = {best}
        for i in range(len(images)):
            if len(shown) >= self.image_limit:
                break
            shown.add(i)
        for i, score in enumerate(rewards):
            media = self.wandb.Image(images[i], caption=f"{method} | particle {i} | reward {score:.5f}") if i in shown else None
            table.add_data(i, score, i == best, *[scores[i] for scores in evaluation.values()], media)
        data = {f"{method}/final/particles": table,
                f"{method}/final/best_image": self.wandb.Image(images[best], caption=f"{method} | reward {rewards[best]:.5f}"),
                f"{method}/final/reward_histogram": self.wandb.Histogram(rewards)}
        self.run.log(data)
        artifact = self.wandb.Artifact(f"results-{self.run.id}-{method}", type="sampling-results")
        artifact.add_file(str(Path(directory) / "metrics.json"))
        artifact.add_file(str(Path(directory) / "best.png"))
        self.run.log_artifact(artifact)

    def log_comparison(self):
        if self.run is None or len(self.completed) < 2:
            return
        table = self.wandb.Table(columns=["method", "mean_reward", "best_reward", "wall_seconds",
                                           "denoiser_batch_elements", "total_denoiser_batch_elements",
                                           "reward_images"])
        for method, values in self.completed.items():
            table.add_data(method, values["reward_mean"], values["reward_max"], values["wall_seconds"],
                           values["denoiser_batch_elements"], values["total_denoiser_batch_elements"],
                           values["reward_image_evaluations"])
            if "unguided" in self.completed:
                for metric in ("reward_mean", "reward_max"):
                    self.run.summary[f"{method}/vs_unguided/{metric}_delta"] = (
                        values[metric] - self.completed["unguided"][metric])
        self.run.log({"comparison/results": table,
                      "comparison/mean_reward": self.wandb.plot.bar(table, "method", "mean_reward", title="Mean final reward"),
                      "comparison/best_reward": self.wandb.plot.bar(table, "method", "best_reward", title="Best final reward")})
        for name in dict.fromkeys(name for metrics in self.evaluations.values() for name in metrics):
            table = self.wandb.Table(columns=["method", "mean", "at_selected_particle"])
            for method, metrics in self.evaluations.items():
                if name in metrics:
                    table.add_data(method, metrics[name]["mean"], metrics[name]["at_selected_particle"])
            self.run.log({f"comparison/{name}/scores": table,
                          f"comparison/{name}/mean": self.wandb.plot.bar(table, "method", "mean", title=f"{name}: mean reward"),
                          f"comparison/{name}/selected": self.wandb.plot.bar(table, "method", "at_selected_particle", title=f"{name}: guidance-selected image")})

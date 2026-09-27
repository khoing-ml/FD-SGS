import sys
from unittest.mock import MagicMock

import pytest

from fd_sgs.tracking import WandbLogger


def make_logger(tmp_path, **kwargs):
    return WandbLogger(mode="offline", project="test", config={"seed": 42}, directory=tmp_path, **kwargs)


def test_disabled_tracking_does_not_import_wandb(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with WandbLogger(mode="disabled", project="test", config={}, directory=tmp_path) as logger:
        assert logger.step_callback("fd-sgs") is None
        logger.log_result("fd-sgs", [], {}, tmp_path)
        logger.log_comparison()


@pytest.mark.parametrize("fail", [False, True])
def test_run_is_finished_on_success_and_failure(monkeypatch, tmp_path, fail):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    def experiment():
        with make_logger(tmp_path):
            if fail:
                raise RuntimeError("generation failed")
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            experiment()
    else:
        experiment()
    sdk.init.return_value.finish.assert_called_once_with(exit_code=int(fail))
    assert sdk.init.call_args.kwargs["config"] == {"seed": 42}


def test_step_logging_pairs_probe_rewards_and_uses_method_axes(monkeypatch, tmp_path):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    with make_logger(tmp_path, previews=True, image_limit=1) as logger:
        for method in ("independent-fd", "fd-sgs"):
            callback = logger.step_callback(method)
            callback({"step": 3, "sigma": .8, "sigma_next": .6, "guided": True,
                      "anchor_rewards": [1., 3.], "twin_rewards": [2., 5., 4., 7.],
                      "correction_ratios": [.01, .03]}, list(range(6)))
            data = logger.run.log.call_args.args[0]
            assert data[f"{method}/step"] == 3
            assert data[f"{method}/sampling/reward_difference_mean"] == 2.5
            assert data[f"{method}/sampling/correction_ratio_max"] == .03
            assert len(data[f"{method}/sampling/clean_predictions"]) == 2
            assert "step" not in logger.run.log.call_args.kwargs
        assert logger.run.define_metric.call_count == 4


def test_final_tables_summaries_comparison_and_image_cap(monkeypatch, tmp_path):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    stats = dict(final_rewards=[1., 3., 2.], best_particle=1, sequential_nfe=8,
                 denoiser_forward_calls=8, denoiser_batch_elements=48, reward_calls=4,
                 reward_image_evaluations=21, guidance_reward_image_evaluations=18,
                 particles=3, twin_trajectories=3, probes_per_particle=1, wall_seconds=2.,
                 peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0)
    with make_logger(tmp_path, image_limit=1) as logger:
        logger.log_result("unguided", ["a", "b", "c"], stats, tmp_path)
        summary = logger.run.summary.update.call_args.args[0]
        assert summary["unguided/final/reward_mean"] == 2.
        assert summary["unguided/final/reward_max"] == 3.
        rows = sdk.Table.return_value.add_data.call_args_list
        assert rows[0].args[-1] is None
        assert rows[1].args[-1] is not None  # winner survives the cap
        assert rows[2].args[-1] is None
        logger.log_result("fd-sgs", ["a", "b", "c"], {**stats, "final_rewards": [2., 4., 3.]}, tmp_path)
        logger.log_comparison()
        logger.run.summary.__setitem__.assert_any_call("fd-sgs/vs_unguided/reward_mean_delta", 1.)
        assert sdk.plot.bar.call_count == 2
        assert logger.run.log_artifact.call_count == 2

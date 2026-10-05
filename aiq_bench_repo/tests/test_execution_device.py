from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from architecture_iq.candidates.axes import choices_compatible
from architecture_iq.candidates.generator import (
    CLASSIFICATION_TRAIN_PY,
    build_candidate_spec,
    _process_train_py,
    validate_process_request,
)
from architecture_iq.ground_truth.runner import _resolve_execution_device
from architecture_iq.profile import load_profile


def _spec(*, device: str, width: int = 16) -> dict:
    profile = load_profile("v1")
    return build_candidate_spec(
        profile,
        dataset_id="sym_device_test",
        family="univariate_regression",
        budget=1024,
        batch_size=16,
        model={
            "type": "mlp",
            "depth": 2,
            "width": width,
            "residual": False,
            "layer_norm": [False, False],
            "activation": "relu",
        },
        optimizer={
            "type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        loss={"loss_id": "mse"},
        execution_device=device,
    )


def test_frozen_v1_profile_keeps_legacy_hash_and_cpu_default() -> None:
    profile = load_profile("v1")
    assert "device" not in profile.ground_truth
    assert profile.execution_device == "cpu"
    assert profile.profile_hash == "90421abe32ec88c7"


def test_legacy_candidate_without_execution_is_always_cpu() -> None:
    profile = load_profile("v1")
    profile.ground_truth["device"] = "cuda"
    assert str(_resolve_execution_device({}, profile)) == "cpu"


def test_device_changes_candidate_identity_and_mixed_choices_are_rejected() -> None:
    cpu = _spec(device="cpu", width=16)
    cuda = _spec(device="cuda", width=32)
    assert cpu["candidate_id"] != cuda["candidate_id"]
    assert not choices_compatible([cpu, cuda])


def test_classification_generated_train_loop_is_device_aware() -> None:
    assert 'device: str = "cpu"' in CLASSIFICATION_TRAIN_PY
    assert "Model().to(run_device)" in CLASSIFICATION_TRAIN_PY
    assert "device=run_device" in CLASSIFICATION_TRAIN_PY


def _process_spec() -> dict:
    return {"family": "synthetic_tabular_classification", "execution": {"device": "cpu"},
            "model": {"type": "mlp", "input_dim": 2, "output_dim": 2, "depth": 1, "width": 4,
                      "activation": "relu", "layer_norm": [False], "residual": False},
            "optimizer": {"type": "Adam", "lr": .01, "weight_decay": .1, "betas": [.9, .999]},
            "loss": {"loss_id": "cross_entropy"}, "budget": {"training_steps": 3, "batch_size": 2}}


def test_process_request_normalizes_and_rejects_unsupported_controls() -> None:
    spec = _process_spec()
    normalized = validate_process_request({"steps": [3, 1]}, spec)
    assert normalized == {"version": "adam_ce_v1", "steps": [1, 3], "max_eval_samples": 4096}
    for request in ({}, None, {"steps": [True]}, {"steps": [0]}, {"steps": [4]}, {"steps": [1, 1]},
                    {"steps": [1], "max_eval_samples": 4097}, {"steps": [1], "unknown": True}):
        with pytest.raises(ValueError):
            validate_process_request(request, spec)
    overrides = [("family", "multivariate_regression"), ("execution", {"device": "cuda"}),
                 ("loss", {"loss_id": "cross_entropy_l2", "lambda": .01}),
                 ("optimizer", dict(spec["optimizer"], type="AdamW")),
                 ("optimizer", dict(spec["optimizer"], weight_decay_scope="head")),
                 ("optimizer", dict(spec["optimizer"], eps=1e-6)),
                 ("optimizer", dict(spec["optimizer"], betas=[1, .999])),
                 ("model", dict(spec["model"], init="custom"))]
    for key, value in overrides:
        invalid = copy.deepcopy(spec)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_process_request({"steps": [1]}, invalid)


def test_process_norms_angles_inactivity_and_coupled_adam_direction() -> None:
    import architecture_iq.candidates.generator as generator
    namespace = {"torch": torch, "F": torch.nn.functional, "hashlib": __import__("hashlib")}
    exec(generator.PROCESS_HELPERS, namespace)
    model = torch.nn.Module()
    model.net = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.)
        model.net[0].weight.copy_(torch.tensor([[3., 4.]]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.1, betas=(.9, .999))
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.net[0].weight.grad.copy_(torch.tensor([[0., 2.]]))
    model.net[0].bias.grad = None
    tensors = (torch.zeros(2, 2), torch.zeros(2, dtype=torch.long),
               torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
    recorder = namespace["_ProcessRecorder"](model, optimizer,
        {"version": "adam_ce_v1", "steps": [1], "max_eval_samples": 2}, 7, tensors)
    snapshot = recorder.before(1, torch.tensor([0, 1]))
    optimizer.step()
    recorder.after(snapshot)
    rows = {row["name"]: row for row in recorder.steps[0]["parameters"]}
    row = rows["net.0.weight"]
    assert row["parameter_norm"] == 5
    assert row["data_gradient_norm"] == 2
    assert row["decay_gradient_norm"] == pytest.approx(.5)
    assert row["data_decay_cosine"]["value"] == pytest.approx(.8)
    assert row["decay_to_data_ratio"]["value"] == pytest.approx(.25)
    assert row["coupled_gradient_norm"] == pytest.approx(math.sqrt(.3**2 + 2.4**2))
    assert row["moment_after"]["exp_avg"] == pytest.approx(.1 * math.sqrt(.3**2 + 2.4**2))
    assert row["preconditioned_direction_norm"] == pytest.approx(math.sqrt(2), rel=1e-6)
    assert row["descent_norm"] == pytest.approx(.01 * math.sqrt(2), rel=1e-4)
    assert row["adam_step_check"]["passed"]
    inactive = rows["net.0.bias"]
    assert not inactive["active"] and inactive["descent_norm"] == 0
    zero = rows["net.1.weight"]
    assert zero["decay_to_data_ratio"]["value"] is None
    assert zero["data_decay_cosine"]["value"] is None
    assert zero["role"] == "head" and row["role"] == "hidden"
    assert namespace["_ratio"](0., 0.)["reason"] == "zero denominator"
    assert namespace["_cosine"](torch.zeros(2), torch.ones(2))["reason"] == "zero vector"


def test_process_runner_rejects_bound_and_malformed_return_before_saving(tmp_path) -> None:
    import architecture_iq.ground_truth.runner as runner
    spec = _process_spec()
    spec["process"] = validate_process_request({"steps": [1], "max_eval_samples": 2}, spec)
    x, y = torch.zeros(3, 2), torch.zeros(3, dtype=torch.long)
    with patch.object(runner, "load_candidate_train") as loader:
        with pytest.raises(ValueError, match="sample count"):
            runner.run_single_seed(tmp_path, spec, x, y, x, y, 7, float("inf"),
                                   selection_metric="test_ce", device=torch.device("cpu"))
        loader.assert_not_called()
    def train_and_eval(*args, process, **kwargs):
        return {"failed": False, "final_test_ce": .7, "final_test_accuracy": .5,
                "eval_samples": [2], "step_metrics": [.7], "process_diagnostics": {"record": {}, "evaluation": {}}}
    with patch.object(runner, "load_candidate_train", return_value=SimpleNamespace(train_and_eval=train_and_eval)):
        with pytest.raises(ValueError, match="record identity"):
            runner.run_single_seed(tmp_path, spec, x[:2], y[:2], x[:2], y[:2], 7, float("inf"),
                                   selection_metric="test_ce", device=torch.device("cpu"))

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kb_lab
sys.path.insert(0, str(kb_lab.BENCH / "src"))
from architecture_iq.candidates.generator import write_candidate, validate_process_request
from architecture_iq.ground_truth.runner import run_ground_truth, run_single_seed, validate_target_transform
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type


def spec(optimizer):
    return {"schema_version": "1.0", "family": "multivariate_regression", "dataset_id": "fixture",
            "candidate_id": "fixture", "execution": {"device": "cpu"},
            "model": {"type": "mlp", "input_dim": 2, "output_dim": 1, "depth": 2, "width": 6,
                      "activation": "gelu", "layer_norm": [False, True], "residual": False},
            "optimizer": optimizer, "loss": {"loss_id": "mse"},
            "budget": {"training_steps": 4, "batch_size": 3, "total_samples_seen": 12}}


def tensors():
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    return x, (x[:, :1] ** 2 + 2 * x[:, 1:] + 3), x + .1, (x[:, :1] ** 2 - x[:, 1:] + 2)


@pytest.mark.parametrize("optimizer", [
    {"type": "Adam", "lr": .001, "weight_decay": .001, "betas": [.9, .999]},
    {"type": "SGD", "lr": .001, "weight_decay": .001, "momentum": 0.0},
    {"type": "SGD", "lr": .001, "weight_decay": .001, "momentum": .9},
])
def test_recorder_preserves_training_rng_and_reads_back_metrics(tmp_path, optimizer):
    ensure_registries()
    torch.set_num_threads(1)
    plain = spec(optimizer)
    recorded = copy.deepcopy(plain)
    recorded["process"] = validate_process_request({"steps": [4, 1], "max_eval_samples": 4}, recorded)
    write_candidate(plain, tmp_path / "plain", get_model_type("mlp"))
    write_candidate(recorded, tmp_path / "recorded", get_model_type("mlp"))
    data = tensors()
    baseline = run_single_seed(tmp_path / "plain", plain, *data, 7, float("inf"),
                               selection_metric="test_mse", device=torch.device("cpu"))
    rng = torch.get_rng_state().clone()
    result = run_single_seed(tmp_path / "recorded", recorded, *data, 7, float("inf"),
                            selection_metric="test_mse", device=torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    assert {k: v for k, v in result.items() if k != "process_diagnostics"} == baseline
    d = result["process_diagnostics"]
    assert [row["step"] for row in d["record"]["steps"]] == [1, 4]
    assert d["record"]["initial_metrics"]["train_target_mean"] == 4
    assert d["record"]["initial_parameters"]
    assert all(p["descent_norm"] >= 0 and p["displacement_norm"] >= 0
               for row in d["record"]["steps"] for p in row["parameters"])
    arrays = d["evaluation"]
    assert float(((arrays["predictions"] - arrays["targets"]) ** 2).mean()) == baseline["final_test_mse"]


def test_target_transform_matches_manual_labels_and_preserves_inputs(tmp_path):
    ensure_registries()
    torch.set_num_threads(1)
    plain = spec({"type": "SGD", "lr": .001, "momentum": .9})
    transformed = copy.deepcopy(plain)
    transformed["target_transform"] = validate_target_transform({"center": True, "offset": 1.5}, transformed)
    write_candidate(plain, tmp_path / "plain", get_model_type("mlp"))
    write_candidate(transformed, tmp_path / "shifted", get_model_type("mlp"))
    tx, ty, vx, vy = tensors()
    frozen = [t.clone() for t in (tx, ty, vx, vy)]
    shift = 1.5 - float(ty.double().mean())
    expected = run_single_seed(tmp_path / "plain", plain, tx, ty + shift, vx, vy + shift, 7, float("inf"),
                               selection_metric="test_mse", device=torch.device("cpu"))
    result = run_single_seed(tmp_path / "shifted", transformed, tx, ty, vx, vy, 7, float("inf"),
                            selection_metric="test_mse", device=torch.device("cpu"))
    assert {k: v for k, v in result.items() if k != "target_transform"} == expected
    assert all(torch.equal(old, new) for old, new in zip(frozen, (tx, ty, vx, vy)))
    assert result["target_transform"]["transformed_labels"]["train_y"]["mean"] == 1.5
    assert result["target_transform"]["transformed_labels"]["test_y"]["mean"] == .5
    for request in ({"offset": float("nan")}, {"offset": True}, {"center": "true"}, {"arbitrary": 1}):
        with pytest.raises(ValueError):
            validate_target_transform(request, plain)


def test_lab_transform_cache_and_source_roundtrip(tmp_path):
    ensure_registries()
    data = tensors()
    datasets = tmp_path / "datasets"
    directory = datasets / "multivariate_regression/fixture"
    directory.mkdir(parents=True)
    torch.save({"x": data[0], "y": data[1]}, directory / "train.pt")
    torch.save({"x": data[2], "y": data[3]}, directory / "test.pt")
    (directory / "dataset_spec.json").write_text(json.dumps({"family": "multivariate_regression",
        "dataset_id": "fixture", "selection_metric": "test_mse", "params": {}, "significance": {}}))
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    candidate = spec({"type": "Adam", "lr": .001})
    candidate = {k: candidate[k] for k in ("model", "optimizer", "loss", "budget")}
    job = {"dataset": "multivariate_regression/fixture", "candidates": [candidate],
           "capture_executable_sources": True, "target_transform": {"center": True, "offset": 3},
           "process": {"steps": [1, 4], "max_eval_samples": 4},
           "diagnostic_fail_threshold": 1e9}
    profile = load_profile("v1.5")
    profile.ground_truth.update(n_seeds=1, base_seed=7, fail_threshold=1e9)
    with patch.object(kb_lab, "DATASETS", datasets), \
         patch("architecture_iq.profile.load_profile", return_value=profile), \
         patch.object(kb_lab.cf, "ProcessPoolExecutor", ThreadPoolExecutor), \
         patch.dict(kb_lab.os.environ, {"ARCHITECTURE_IQ_TORCH_THREADS": "1", "ARCHITECTURE_IQ_SEED_WORKERS": "1"}):
        fresh = kb_lab.run_job(job, workers=1, lab_dir=tmp_path / "lab")["results"][0]
        cached = kb_lab.run_job(job, workers=1, lab_dir=tmp_path / "lab")["results"][0]
        changed = kb_lab.run_job({**job, "target_transform": {"center": True, "offset": 0}},
                                 workers=1, lab_dir=tmp_path / "lab")["results"][0]
    assert not fresh["cached"] and cached["cached"] and not changed["cached"]
    assert fresh["failed_seeds"] == 0
    assert fresh["source_provenance"]["status"] == "verified"
    assert fresh["process_provenance"]["recorder_version"] == "regression_mse_v1"
    assert fresh["seed_results"][0]["target_transform"]["transformed_labels"]["train_y"]["mean"] == 3
    assert "executed/target_transform_runner.py" in fresh["measurement_files"]
    assert pins == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    summary = Path(fresh["measurement_files"]["results/summary.json"]["path"])
    with np.load(summary.parent / "process/seed_7.npz") as arrays:
        assert arrays["targets"].shape == data[3].shape
        assert abs(float(arrays["train_targets"].mean()) - 3) < 1e-6

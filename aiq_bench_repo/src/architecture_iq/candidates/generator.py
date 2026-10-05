from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from architecture_iq.losses import render_loss_py
from architecture_iq.models.base import ModelFamily
from architecture_iq.candidates.axes import choices_compatible as choices_compatible
from architecture_iq.optimizers.factory import render_optimizer_py
from architecture_iq.profile import Profile, validate_execution_device
from architecture_iq.registry import (
    ensure_registries,
    get_dataset_family,
    get_model_type,
)
from architecture_iq.util import short_hash, write_json

REGRESSION_TRAIN_PY = '''"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_mse(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        pred = model(test_x)
        return float(torch.mean((pred - test_y) ** 2).item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        pred = model(train_x[idx])
        loss = loss_fn(model, pred, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric = _test_mse(model, test_x, test_y)
        if not math.isfinite(metric):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(metric)
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": metric,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "failed": failed,
        "final_test_mse": final_metric,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }


def train(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Minimal training entrypoint (no evaluation)."""
    train_and_eval(
        train_x,
        train_y,
        test_x=train_x,
        test_y=train_y,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        fail_threshold=float("inf"),
        device=device,
    )
'''

LM_TRAIN_PY = '''"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_ce(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        pred = model(test_x)
        if pred.ndim == 3:
            vocab = pred.shape[-1]
            loss = F.cross_entropy(pred.reshape(-1, vocab), test_y.reshape(-1))
        else:
            loss = F.cross_entropy(pred, test_y.reshape(-1))
        return float(loss.item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        pred = model(train_x[idx])
        loss = loss_fn(model, pred, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric = _test_ce(model, test_x, test_y)
        if not math.isfinite(metric):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(metric)
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": metric,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "failed": failed,
        "final_test_ce": final_metric,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }
'''


CLASSIFICATION_TRAIN_PY = '''"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_metrics(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(test_x)
        ce = F.cross_entropy(logits, test_y.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == test_y.reshape(-1)).float().mean()
    return float(ce.item()), float(accuracy.item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)
    final_accuracy = float("nan")

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        logits = model(train_x[idx])
        loss = loss_fn(model, logits, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ce, accuracy = _test_metrics(model, test_x, test_y)
        if not math.isfinite(ce) or not math.isfinite(accuracy):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(ce)
        final_accuracy = accuracy
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": ce,
                    "accuracy": accuracy,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "failed": failed,
        "final_test_ce": final_metric,
        "final_test_accuracy": final_accuracy,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }
'''


PROCESS_RECORDER_VERSION = "adam_ce_v1"
PROCESS_MAX_STEPS = 16
PROCESS_MAX_EVAL = 4096


def validate_process_request(request: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    ensure_registries()
    if get_dataset_family(spec["family"]).train_loop_kind == "regression":
        return _validate_regression_process(request, spec)
    if not isinstance(request, dict) or set(request) - {"steps", "max_eval_samples", "version"}:
        raise ValueError("process request accepts only steps, max_eval_samples and version")
    if request.get("version", PROCESS_RECORDER_VERSION) != PROCESS_RECORDER_VERSION:
        raise ValueError("unsupported process recorder version")
    ensure_registries()
    if get_dataset_family(spec["family"]).train_loop_kind != "classification":
        raise ValueError("process recording supports classification only")
    if set(spec.get("execution", {})) - {"device"}:
        raise ValueError("unsupported process execution control")
    if spec.get("execution", {}).get("device", "cpu") != "cpu" or spec["model"].get("type") != "mlp":
        raise ValueError("process recording supports CPU MLP only")
    if spec["loss"] != {"loss_id": "cross_entropy"}:
        raise ValueError("process recording requires plain cross_entropy without extra controls")
    optimizer = spec["optimizer"]
    if optimizer.get("type") != "Adam" or set(optimizer) - {"type", "lr", "weight_decay", "betas"}:
        raise ValueError("process recording supports coupled Adam with type/lr/weight_decay/betas only")
    betas = optimizer.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
        isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1 for b in betas
    ):
        raise ValueError("process Adam betas must be two finite values in [0,1)")
    for key, default, positive in (("lr", None, True), ("weight_decay", 0.0, False)):
        value = optimizer.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"process Adam {key} is invalid")
    model = spec["model"]
    known_model = {"type", "depth", "width", "input_dim", "output_dim", "activation", "activations",
                   "layer_norm", "residual", "leaky_relu_slope"}
    if set(model) - known_model:
        raise ValueError("unsupported process model control")
    if "activation" in model and "activations" in model:
        raise ValueError("process model must use one activation representation")
    for key in ("depth", "width", "input_dim"):
        if isinstance(model.get(key), bool) or not isinstance(model.get(key), int) or model[key] < 1:
            raise ValueError(f"process model {key} must be a positive integer")
    if not isinstance(model.get("residual"), bool) or not isinstance(model.get("layer_norm"), list) or len(model["layer_norm"]) != model["depth"] or any(not isinstance(v, bool) for v in model["layer_norm"]):
        raise ValueError("process MLP residual and depth-matched layer_norm must be boolean")
    if "leaky_relu_slope" in model:
        value = model["leaky_relu_slope"]
        if model.get("activation") != "leaky_relu" or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("unsupported process leaky_relu_slope")
    classes = model.get("output_dim", 1)
    if isinstance(classes, bool) or not isinstance(classes, int) or not 2 <= classes <= 256:
        raise ValueError("process output_dim must be an integer from 2 to 256")
    horizon = spec["budget"]["training_steps"]
    if set(spec["budget"]) - {"training_steps", "batch_size", "total_samples_seen"}:
        raise ValueError("unsupported process budget control")
    batch = spec["budget"].get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("process batch_size must be a positive integer")
    steps = request.get("steps")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("process training_steps must be a positive integer")
    if "total_samples_seen" in spec["budget"] and spec["budget"]["total_samples_seen"] != horizon * batch:
        raise ValueError("process budget sample count mismatch")
    if not isinstance(steps, list) or not 1 <= len(steps) <= PROCESS_MAX_STEPS or any(
        isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= horizon for s in steps
    ) or len(set(steps)) != len(steps):
        raise ValueError("process steps must contain 1 to 16 distinct integers inside the training horizon")
    maximum = request.get("max_eval_samples", PROCESS_MAX_EVAL)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= PROCESS_MAX_EVAL:
        raise ValueError("process max_eval_samples must be an integer from 1 to 4096")
    return {"version": PROCESS_RECORDER_VERSION, "steps": sorted(steps), "max_eval_samples": maximum}


def _validate_regression_process(request, spec):
    version = "regression_mse_v1"
    if not isinstance(request, dict) or set(request) - {"steps", "max_eval_samples", "version"} or request.get("version", version) != version:
        raise ValueError("unsupported regression process request")
    if spec.get("execution", {"device": "cpu"}) != {"device": "cpu"} or spec["model"].get("type") != "mlp" or spec["loss"] != {"loss_id": "mse"}:
        raise ValueError("regression process requires CPU MLP and plain MSE")
    model = spec["model"]
    known = {"type", "depth", "width", "input_dim", "output_dim", "activation", "activations",
             "layer_norm", "residual", "leaky_relu_slope"}
    if set(model) - known or ("activation" in model and "activations" in model):
        raise ValueError("unsupported regression process model control")
    if any(isinstance(model.get(key), bool) or not isinstance(model.get(key), int) or model[key] < 1 for key in ("depth", "width", "input_dim")) or model.get("output_dim", 1) != 1:
        raise ValueError("regression process requires positive model sizes and scalar output")
    if not isinstance(model.get("residual"), bool) or not isinstance(model.get("layer_norm"), list) or len(model["layer_norm"]) != model["depth"] or any(not isinstance(v, bool) for v in model["layer_norm"]):
        raise ValueError("regression process requires boolean residual and depth-matched LN")
    opt = spec["optimizer"]
    allowed = {"type", "lr", "weight_decay", "betas"} if opt.get("type") == "Adam" else {"type", "lr", "weight_decay", "momentum"}
    if opt.get("type") not in {"Adam", "SGD"} or set(opt) - allowed:
        raise ValueError("regression process supports canonical Adam or SGD")
    for key, default in (("lr", None), ("weight_decay", 0.0)):
        value = opt.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            raise ValueError("invalid regression process optimizer value")
    values = opt.get("betas", [.9, .999]) if opt["type"] == "Adam" else [opt.get("momentum", 0.0)]
    if not isinstance(values, (list, tuple)) or len(values) != (2 if opt["type"] == "Adam" else 1) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1 for v in values):
        raise ValueError("invalid regression process momentum")
    budget = spec["budget"]
    if set(budget) - {"training_steps", "batch_size", "total_samples_seen"} or any(isinstance(budget.get(key), bool) or not isinstance(budget.get(key), int) or budget[key] < 1 for key in ("training_steps", "batch_size")):
        raise ValueError("invalid regression process budget")
    if "total_samples_seen" in budget and budget["total_samples_seen"] != budget["training_steps"] * budget["batch_size"]:
        raise ValueError("regression process sample count mismatch")
    steps, maximum = request.get("steps"), request.get("max_eval_samples", PROCESS_MAX_EVAL)
    if not isinstance(steps, list) or not 1 <= len(steps) <= PROCESS_MAX_STEPS or any(isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= budget["training_steps"] for s in steps) or len(set(steps)) != len(steps):
        raise ValueError("invalid regression observation steps")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= PROCESS_MAX_EVAL:
        raise ValueError("invalid regression evaluation bound")
    return {"version": version, "steps": sorted(steps), "max_eval_samples": maximum}


PROCESS_HELPERS = r'''
def _tensor_pin(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def _norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def _ratio(numerator, denominator):
    if denominator == 0:
        return {"value": None, "reason": "zero denominator"}
    return {"value": numerator / denominator, "reason": None}


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return {"value": None, "reason": "zero vector"}
    return {"value": float(torch.sum(left.double() * right.double()).item()) / denominator, "reason": None}


class _ProcessRecorder:
    def __init__(self, model, optimizer, request, seed, tensors):
        if request is None or request.get("version") != "adam_ce_v1":
            raise ValueError("recording requires a normalized adam_ce_v1 request")
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise ValueError("recording requires canonical single-group coupled Adam")
        self.group = optimizer.param_groups[0]
        if any(self.group.get(k, False) for k in ("amsgrad", "maximize", "capturable", "differentiable", "fused", "decoupled_weight_decay")):
            raise ValueError("unsupported process Adam mode")
        if tensors[2].shape[0] > request["max_eval_samples"] or tensors[2].shape[0] == 0:
            raise ValueError("lab evaluation sample count exceeds requested bound or is empty")
        self.named = list(model.named_parameters())
        if len(self.named) > 256:
            raise ValueError("process parameter count exceeds recorder bound")
        self.head = f"net.{len(model.net) - 1}."
        self.optimizer, self.request, self.seed = optimizer, request, seed
        self.stream = hashlib.sha256()
        self.steps = []
        self.record = {"schema_version": 1, "recorder_version": "adam_ce_v1", "request": request,
                       "seed": seed, "optimizer_type": "Adam", "decay_mode": "coupled",
                       "optimizer_defaults": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in self.group.items() if k != "params"},
                       "inputs": {name: _tensor_pin(value) for name, value in zip(
                       ("train_x", "train_y", "test_x", "test_y"), tensors)}, "steps": self.steps}

    def before(self, step, indices):
        self.stream.update(indices.detach().cpu().contiguous().numpy().tobytes())
        if step not in self.request["steps"]:
            return None
        snapshot = []
        for name, parameter in self.named:
            weight = parameter.detach().clone()
            gradient = parameter.grad.detach().clone() if parameter.grad is not None else None
            state = self.optimizer.state.get(parameter, {})
            entry = {"name": name, "role": "head" if name.startswith(self.head) else "hidden",
                     "shape": list(weight.shape), "active": gradient is not None, "parameter_norm": _norm(weight),
                     "moment_before": {key: _norm(state[key]) if key in state else 0.0
                                       for key in ("exp_avg", "exp_avg_sq")}}
            if gradient is None:
                entry["reason"] = "gradient absent; Adam skips this parameter"
            else:
                decay = self.group["weight_decay"] * weight
                entry.update(data_gradient_norm=_norm(gradient), decay_gradient_norm=_norm(decay),
                             coupled_gradient_norm=_norm(gradient + decay),
                             decay_to_data_ratio=_ratio(_norm(decay), _norm(gradient)),
                             parameter_data_cosine=_cosine(weight, gradient), parameter_decay_cosine=_cosine(weight, decay),
                             data_decay_cosine=_cosine(gradient, decay))
            snapshot.append((parameter, weight, gradient, entry))
        return {"step": step, "minibatch": _tensor_pin(indices), "parameters": snapshot}

    def after(self, snapshot):
        if snapshot is None:
            return
        rows = []
        for parameter, weight, gradient, entry in snapshot["parameters"]:
            descent = weight - parameter.detach()
            entry.update(descent_norm=_norm(descent), relative_descent=_ratio(_norm(descent), _norm(weight)))
            state = self.optimizer.state.get(parameter, {})
            entry["moment_after"] = {key: _norm(state[key]) if key in state else 0.0 for key in ("exp_avg", "exp_avg_sq")}
            if gradient is not None:
                beta1, beta2 = self.group["betas"]
                count = float(state["step"])
                direction = (state["exp_avg"] / (1 - beta1 ** count)) / (
                    torch.sqrt(state["exp_avg_sq"] / (1 - beta2 ** count)) + self.group["eps"])
                predicted = self.group["lr"] * direction
                tolerance = 8 * torch.finfo(weight.dtype).eps * max(1.0, float(weight.abs().max()))
                valid = bool(torch.allclose(descent, predicted, rtol=5e-5, atol=tolerance))
                entry.update(preconditioned_direction_norm=_norm(direction), predicted_descent_norm=_norm(predicted),
                             preconditioned_data_cosine=_cosine(direction, gradient),
                             preconditioned_decay_cosine=_cosine(direction, self.group["weight_decay"] * weight),
                             preconditioned_parameter_cosine=_cosine(direction, weight),
                             descent_data_cosine=_cosine(descent, gradient),
                             descent_decay_cosine=_cosine(descent, self.group["weight_decay"] * weight),
                             descent_parameter_cosine=_cosine(descent, weight),
                             adam_step_check={"passed": valid, "error_norm": _norm(descent - predicted),
                                              "atol": tolerance, "rtol": 5e-5}, moment_step=count)
                if not valid:
                    raise ValueError("recorded Adam direction does not match actual descent")
            elif torch.count_nonzero(descent).item():
                raise ValueError("inactive Adam parameter unexpectedly moved")
            rows.append(entry)
        self.steps.append({"step": snapshot["step"], "minibatch": snapshot["minibatch"], "parameters": rows})

    def finish(self, model, test_x, test_y, failed):
        self.record["status"] = "failed" if failed else "completed"
        self.record["minibatch_stream_sha256"] = self.stream.hexdigest()
        if failed:
            self.record["final_metrics"] = None
            return {"record": self.record, "evaluation": {}}
        model.eval()
        with torch.inference_mode():
            logits = model(test_x).detach().cpu()
            targets = test_y.reshape(-1).detach().cpu()
            predictions = logits.argmax(dim=-1)
            other = logits.clone()
            other.scatter_(1, targets[:, None], float("-inf"))
            margins = logits.gather(1, targets[:, None]).reshape(-1) - other.max(dim=1).values
            self.record["final_metrics"] = {"ce": float(F.cross_entropy(logits, targets)),
                                            "accuracy": float((predictions == targets).float().mean())}
        return {"record": self.record, "evaluation": {"indices": torch.arange(targets.shape[0]),
                "logits": logits, "targets": targets, "predictions": predictions,
                "errors": predictions != targets, "true_class_margins": margins}}
'''


def _process_train_py() -> str:
    text = CLASSIFICATION_TRAIN_PY.replace("import math\n", "import math\nimport hashlib\n", 1)
    text = text.replace("    progress_callback=None,\n", "    progress_callback=None,\n    process=None,\n", 1)
    text = text.replace("    n = train_x.shape[0]\n", "    recorder = _ProcessRecorder(model, optimizer, process, seed, (train_x, train_y, test_x, test_y))\n    n = train_x.shape[0]\n", 1)
    text = text.replace("        optimizer.step()\n", "        observation = recorder.before(step, idx)\n        optimizer.step()\n        recorder.after(observation)\n", 1)
    text = text.replace("        \"failed\": failed,\n", "        \"process_diagnostics\": recorder.finish(model, test_x, test_y, failed),\n        \"failed\": failed,\n", 1)
    return text + PROCESS_HELPERS


REGRESSION_PROCESS_HELPERS = r'''
class _RegressionRecorder:
    def __init__(self, model, optimizer, request, seed, tensors):
        if request.get("version") != "regression_mse_v1" or type(optimizer) not in (torch.optim.Adam, torch.optim.SGD) or len(optimizer.param_groups) != 1:
            raise ValueError("unsupported regression recorder")
        if any(t.shape[0] > request["max_eval_samples"] or not t.numel() for t in tensors):
            raise ValueError("regression observation inputs exceed bound or are empty")
        self.model, self.optimizer, self.request = model, optimizer, request
        self.tensors, self.named = tensors, list(model.named_parameters())
        self.initial = {name: p.detach().clone() for name, p in self.named}
        self.head = f"net.{len(model.net) - 1}."
        self.stream, self.steps = hashlib.sha256(), []
        self.initial_train, self.initial_test, metrics = self.evaluate()
        self.record = {"schema_version": 1, "recorder_version": request["version"], "request": request,
                       "seed": seed, "optimizer_type": type(optimizer).__name__, "initial_metrics": metrics,
                       "initial_parameters": {name: _tensor_pin(p) for name, p in self.named},
                       "optimizer_defaults": {k: list(v) if isinstance(v, tuple) else v
                                              for k, v in optimizer.param_groups[0].items() if k != "params"},
                       "inputs": {k: _tensor_pin(t) for k, t in zip(("train_x", "train_y", "test_x", "test_y"), tensors)},
                       "steps": self.steps}

    def evaluate(self):
        was_training = self.model.training
        self.model.eval()
        tx, ty, vx, vy = self.tensors
        with torch.inference_mode():
            train, test = self.model(tx).detach().clone(), self.model(vx).detach().clone()
            metrics = {"train_mse": float(((train - ty) ** 2).mean()),
                       "test_mse": float(((test - vy) ** 2).mean()),
                       "train_prediction_mean": float(train.double().mean()),
                       "test_prediction_mean": float(test.double().mean()),
                       "train_target_mean": float(ty.double().mean()),
                       "test_target_mean": float(vy.double().mean())}
        self.model.train(was_training)
        return train, test, metrics

    def before(self, step, indices):
        self.stream.update(indices.detach().cpu().contiguous().numpy().tobytes())
        if step not in self.request["steps"]:
            return None
        rows = []
        for name, p in self.named:
            weight = p.detach().clone()
            gradient = p.grad.detach() if p.grad is not None else None
            row = {"name": name, "role": "head" if name.startswith(self.head) else "hidden",
                   "active": gradient is not None, "parameter_norm": _norm(weight)}
            if gradient is not None:
                row["data_gradient_norm"] = _norm(gradient)
            rows.append((name, p, weight, row))
        return {"step": step, "minibatch": _tensor_pin(indices), "rows": rows}

    def after(self, snapshot):
        if snapshot is None:
            return
        rows = []
        for name, p, weight, row in snapshot["rows"]:
            row.update(descent_norm=_norm(weight - p.detach()),
                       displacement_norm=_norm(p.detach() - self.initial[name]))
            rows.append(row)
        _, _, metrics = self.evaluate()
        self.steps.append({"step": snapshot["step"], "minibatch": snapshot["minibatch"],
                           "parameters": rows, "metrics": metrics})

    def finish(self, model, test_x, test_y, failed):
        self.record["status"] = "failed" if failed else "completed"
        self.record["minibatch_stream_sha256"] = self.stream.hexdigest()
        if failed:
            self.record["final_metrics"] = None
            return {"record": self.record, "evaluation": {}}
        train, test, metrics = self.evaluate()
        self.record["final_metrics"] = dict(metrics, mse=metrics["test_mse"])
        _, train_y, _, test_y = self.tensors
        return {"record": self.record, "evaluation": {"indices": torch.arange(test_y.shape[0]),
                "predictions": test.cpu(), "targets": test_y.detach().cpu(),
                "initial_predictions": self.initial_test.cpu(), "train_predictions": train.cpu(),
                "train_targets": train_y.detach().cpu(), "initial_train_predictions": self.initial_train.cpu()}}
'''


def _regression_process_train_py() -> str:
    text = REGRESSION_TRAIN_PY.replace("import math\n", "import math\nimport hashlib\n", 1)
    text = text.replace("    progress_callback=None,\n", "    progress_callback=None,\n    process=None,\n", 1)
    text = text.replace("    n = train_x.shape[0]\n", "    recorder = _RegressionRecorder(model, optimizer, process, seed, (train_x, train_y, test_x, test_y))\n    n = train_x.shape[0]\n", 1)
    text = text.replace("        optimizer.step()\n", "        observation = recorder.before(step, idx)\n        optimizer.step()\n        recorder.after(observation)\n", 1)
    text = text.replace("        \"failed\": failed,\n", "        \"process_diagnostics\": recorder.finish(model, test_x, test_y, failed),\n        \"failed\": failed,\n", 1)
    common = PROCESS_HELPERS.split("class _ProcessRecorder:", 1)[0]
    return text + common + REGRESSION_PROCESS_HELPERS


_TRAIN_PY_BY_KIND = {
    "regression": REGRESSION_TRAIN_PY,
    "language_model": LM_TRAIN_PY,
    "classification": CLASSIFICATION_TRAIN_PY,
}


def _train_py_for_family(family: str) -> str:
    """Generated train.py for a family, keyed by its declared train_loop_kind.

    Registry lookup rather than a name branch: a new family declares its kind on
    the plugin, so it cannot silently fall through to the regression loop and
    return a metric key its selection_metric_name() never asked for.
    """
    ensure_registries()
    kind = get_dataset_family(family).train_loop_kind
    try:
        return _TRAIN_PY_BY_KIND[kind]
    except KeyError:
        raise ValueError(
            f"Family {family!r} declares train_loop_kind {kind!r}; "
            f"known kinds are {sorted(_TRAIN_PY_BY_KIND)}"
        ) from None

def _spec_json(spec: dict[str, Any], key: str) -> str:
    return json.dumps(spec[key], sort_keys=True)


def _varying_axes_for_question_type(question_type: str) -> frozenset[str]:
    if question_type == "architecture_only":
        return frozenset({"model"})
    if question_type == "optimizer_only":
        return frozenset({"optimizer"})
    if question_type == "loss_only":
        return frozenset({"loss"})
    if question_type == "mixed":
        return frozenset({"model", "optimizer", "loss"})
    raise ValueError(f"Unknown question type: {question_type}")


def candidate_matches_fixed(spec: dict[str, Any], fixed_shared: dict[str, Any]) -> bool:
    for key, value in fixed_shared.items():
        if key == "batch_size":
            if spec["budget"]["batch_size"] != value:
                return False
        elif key in ("model", "optimizer", "loss"):
            if _spec_json(spec, key) != json.dumps(value, sort_keys=True):
                return False
        else:
            raise ValueError(f"Unknown fixed_shared key: {key}")
    return True


def valid_batch_sizes(profile: Profile, budget: int) -> list[int]:
    min_steps = profile.min_training_steps()
    return [
        b
        for b in profile.optimizer_grids["batch_size"]
        if budget % b == 0 and (min_steps is None or budget // b >= min_steps)
    ]


def _pick_batch_size(profile: Profile, budget: int, rng: random.Random) -> int:
    valid = valid_batch_sizes(profile, budget)
    if not valid:
        raise ValueError(f"No batch size divides budget {budget}")
    return rng.choice(valid)


def sample_optimizer(profile: Profile, rng: random.Random) -> dict[str, Any]:
    opt_type = rng.choice(profile.pools["optimizers"])
    spec: dict[str, Any] = {
        "type": opt_type,
        "lr": rng.choice(profile.optimizer_grids["lr"]),
        "weight_decay": rng.choice(profile.optimizer_grids["weight_decay"]),
    }
    if opt_type == "SGD":
        spec["momentum"] = rng.choice(profile.optimizer_grids["sgd_momentum"])
    if opt_type in {"Adam", "AdamW"}:
        # Only consume RNG when the pool holds a real choice: a profile with one
        # fixed pair must keep the sampling stream it had before this was a pool,
        # so its candidate ids stay reproducible.
        pool = profile.adam_betas_pool()
        beta1, beta2 = pool[0] if len(pool) == 1 else rng.choice(pool)
        spec["betas"] = [float(beta1), float(beta2)]
    return spec


# The only losses that are ever sampled: plain MSE for regression, plain
# cross-entropy for classification / LM. An allowlist rather than a blocklist,
# so a profile cannot reintroduce an exotic loss just by naming it in a pool.
SAMPLEABLE_LOSS_IDS = frozenset({"mse", "cross_entropy"})

REGULARIZED_LOSS_IDS = frozenset(
    {"mse_l1", "mse_l2", "cross_entropy_l1", "cross_entropy_l2"}
)


def sample_loss(profile: Profile, family: str, rng: random.Random) -> dict[str, Any]:
    # Regularisation lives solely in optimizer weight_decay: stacking a
    # parameter-wide loss penalty on top of it made the criterion an ambiguous
    # double-regularisation comparison, and a lambda-weighted penalty changes
    # the objective the reported test metric no longer measures. Renderer and
    # formatter branches for the L1/L2 variants stay for legacy artifacts.
    pool = [
        loss_id
        for loss_id in profile.pools["losses"][family]
        if loss_id in SAMPLEABLE_LOSS_IDS
    ]
    if not pool:
        raise ValueError(
            f"loss pool for {family} has no sampleable loss; "
            f"expected at least one of {sorted(SAMPLEABLE_LOSS_IDS)}"
        )
    return {"loss_id": rng.choice(pool)}


def sample_model(
    profile: Profile,
    rng: random.Random,
    *,
    family: str,
    dataset_params: dict[str, Any] | None = None,
    model_type: str | None = None,
    shared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from architecture_iq.registry import get_dataset_family

    family_obj = get_dataset_family(family)
    model_types = profile.model_types_for_family(
        family,
        family_obj.compatible_model_types(),
    )
    if not model_types:
        raise ValueError(f"No compatible model types for family {family!r}")
    if model_type is None:
        model_type = rng.choice(model_types)
    elif model_type not in model_types:
        raise ValueError(
            f"Model type {model_type!r} is not compatible with family {family!r} "
            f"under profile {profile.name!r}"
        )
    return get_model_type(model_type).sample_spec(
        profile, rng, dataset_params=dataset_params, shared=shared
    )


def trainable_parameter_count(model_spec: dict[str, Any]) -> int:
    """Count unique trainable scalar parameters for a frozen model specification."""
    import torch

    ensure_registries()
    model_family = get_model_type(str(model_spec["type"]))
    with torch.random.fork_rng():
        module = model_family.build_module(model_spec)
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))

def build_candidate_spec(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    batch_size: int,
    model: dict[str, Any],
    optimizer: dict[str, Any],
    loss: dict[str, Any],
    execution_device: str | None = None,
) -> dict[str, Any]:
    steps = profile.training_steps(budget, batch_size)
    device = validate_execution_device(execution_device or profile.execution_device)
    # Double-regularisation guard: a loss-side L1/L2 penalty already applies
    # a parameter-wide penalty, so optimizer weight_decay is zeroed to keep
    # the comparison criterion unambiguous (legacy specs may still carry
    # lambda losses; new sampling never produces them). Zeroed as float so
    # the value survives inspector form roundtrips without changing the
    # candidate id hash.
    if (
        str(loss.get("loss_id")) in REGULARIZED_LOSS_IDS
        and float(optimizer.get("weight_decay") or 0.0) != 0.0
    ):
        optimizer = {**optimizer, "weight_decay": 0.0}
    body = {
        "schema_version": profile.schema_version,
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "dataset_id": dataset_id,
        "family": family,
        "budget": {
            "training_steps": steps,
            "batch_size": batch_size,
            "total_samples_seen": budget,
        },
        "model": model,
        "trainable_parameter_count": trainable_parameter_count(model),
        "optimizer": optimizer,
        "loss": loss,
        "execution": {"device": device},
        "files": {
            "model": "model.py",
            "train": "train.py",
            "loss": "loss.py",
            "optimizer": "optimizer.py",
        },
    }
    body["candidate_id"] = f"c_{short_hash(body)}"
    return body


def write_candidate(
    spec: dict[str, Any],
    out_dir: Path,
    model_family: ModelFamily,
) -> Path:
    process = validate_process_request(spec["process"], spec) if "process" in spec else None
    if process is not None and process != spec["process"]:
        raise ValueError("candidate process request must be normalized before hashing")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "candidate_spec.json", spec)
    (out_dir / "model.py").write_text(
        model_family.render_model_py(spec["model"]), encoding="utf-8"
    )
    (out_dir / "loss.py").write_text(
        render_loss_py(spec["loss"]), encoding="utf-8"
    )
    (out_dir / "optimizer.py").write_text(
        render_optimizer_py(spec["optimizer"]), encoding="utf-8"
    )
    train = _train_py_for_family(spec["family"])
    if process is not None:
        train = _regression_process_train_py() if process["version"] == "regression_mse_v1" else _process_train_py()
    (out_dir / "train.py").write_text(train, encoding="utf-8")
    return out_dir


def sample_candidate(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    rng: random.Random,
    fixed: dict[str, Any] | None = None,
    execution_device: str | None = None,
) -> dict[str, Any]:
    fixed = fixed or {}
    batch_size = fixed.get("batch_size") or _pick_batch_size(profile, budget, rng)
    dataset_params = fixed.get("_dataset_params")
    model = fixed.get("model") or sample_model(
        profile, rng, family=family, dataset_params=dataset_params
    )
    optimizer = fixed.get("optimizer") or sample_optimizer(profile, rng)
    loss = fixed.get("loss") or sample_loss(profile, family, rng)
    return build_candidate_spec(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        batch_size=batch_size,
        model=model,
        optimizer=optimizer,
        loss=loss,
        execution_device=execution_device,
    )


def sample_variant_pool(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    question_type: str,
    pool_size: int,
    rng: random.Random,
    fixed_shared: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from architecture_iq.candidates.sets import sample_candidate_set_pool

    varying_axes = _varying_axes_for_question_type(question_type)
    if question_type == "mixed" and fixed_shared is None:
        varying_axes = frozenset({"model", "optimizer", "loss"})
    return sample_candidate_set_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        count=pool_size,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=fixed_shared,
    )


def sample_variants_for_question(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    question_type: str,
    num_choices: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    return sample_variant_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        question_type=question_type,
        pool_size=num_choices,
        rng=rng,
    )

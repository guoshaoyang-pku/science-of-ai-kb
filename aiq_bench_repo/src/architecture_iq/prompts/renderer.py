from __future__ import annotations

import re
from pathlib import Path

from architecture_iq.ground_truth.runner import _sync_candidate_files
from architecture_iq.paths import PROMPTS_DIR, DATA_DIR, dataset_dir
from architecture_iq.profile import load_profile
from architecture_iq.prompts.code_excerpt import (
    excerpt_loss_py,
    excerpt_model_py,
    excerpt_optimizer_py,
    excerpt_synthesize_py,
)
from architecture_iq.prompts.formatters import (
    SINGLE_AXIS_TYPES,
    TABULAR_CLASSIFICATION_FAMILIES,
    format_dataset_protocol,
    format_loss_nl,
    format_mcq_answer_section,
    format_synthetic_tabular_classification_rule,
    format_model_nl,
    format_optimizer_nl,
    format_ranking_answer_section,
    format_ranking_objective,
    format_ranking_protocol,
    format_training_schedule,
)
from architecture_iq.util import read_json


# Supported prompt tasks. "mcq" picks one winning letter; "ranking" asks for the
# full order and prints the shared model/optimizer once (see render_prompt).
TASKS = frozenset({"mcq", "ranking"})


def _read_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _question_total_samples_seen(budget: dict | int) -> int | None:
    if isinstance(budget, int):
        return budget
    if budget.get("mixed"):
        return None
    return int(budget["total_samples_seen"])


def _evaluation_meta(q: dict) -> dict:
    if "evaluation" in q:
        return q["evaluation"]
    profile = load_profile(q.get("profile", "v1.5"))
    return {
        "selection_metric": "test_mse",
        "n_seeds": profile.n_seeds,
        "base_seed": profile.base_seed,
        "device": "cpu",
    }


def _ranking_shared_sections(
    q: dict, artifact_root: Path, question_path: Path
) -> list[str]:
    """Model + optimizer for a ranking question, printed once instead of per choice.

    A ranking question varies the loss only (model / optimizer / batch size /
    budget are invariant), so the per-choice model and optimizer blocks of the MCQ
    layout would be five identical copies -- measured at 46% of the rendered
    prompt. Repeating them also invites a reader to hunt for a difference that is
    not there, so the layout states the shared setup once. The invariant is
    asserted rather than assumed: if a question labelled a ranking question ever
    varies the model or optimizer, the compression would hide it, so fail loudly.
    """
    first = q["choices"][0]
    first_path = artifact_root / first["candidate_path"]
    first_spec = read_json(first_path / "candidate_spec.json")
    for choice in q["choices"][1:]:
        other = read_json(artifact_root / choice["candidate_path"] / "candidate_spec.json")
        if other["model"] != first_spec["model"]:
            raise ValueError(f"{question_path.name}: choices disagree on the model")
        if other["optimizer"] != first_spec["optimizer"]:
            raise ValueError(f"{question_path.name}: choices disagree on the optimizer")
    _sync_candidate_files(first_path, first_spec)
    return [
        "## Shared setup (same for all choices)",
        "",
        "**Model (natural language)**",
        format_model_nl(first_spec["model"]),
        "",
        "**Model code**",
        "```python",
        excerpt_model_py((first_path / "model.py").read_text(encoding="utf-8")),
        "```",
        "",
        "**Optimizer**",
        format_optimizer_nl(first_spec["optimizer"]),
        "",
        "```python",
        excerpt_optimizer_py((first_path / "optimizer.py").read_text(encoding="utf-8")),
        "```",
        "",
    ]


def _gold_order(q: dict) -> list[str] | None:
    """The ranked letters of a ranking question, from `answer` or `correct_order`."""
    gold = q.get("answer")
    if isinstance(gold, str) and gold:
        return [part for part in gold.replace(">", "<").split("<") if part]
    gold = q.get("correct_order")
    return list(gold) if isinstance(gold, list) and gold else None


def render_prompt(
    question_path: Path,
    *,
    dataset_path: Path | None = None,
    artifact_root: Path | None = None,
    task: str = "mcq",
) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown prompt task {task!r}; expected one of {sorted(TASKS)}")
    q = read_json(question_path / "question.json")
    dataset_path = (dataset_path or dataset_dir(q["family"], q["dataset_id"])).resolve()
    artifact_root = (artifact_root or DATA_DIR).resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    params = dataset_spec["params"]
    eval_meta = _evaluation_meta(q)
    selection_metric = eval_meta.get("selection_metric", dataset_spec["selection_metric"])

    header = _read_template("header_ranking.md" if task == "ranking" else "header.md")
    if not header:
        header = (
            "You are taking the ArchitectureIQ benchmark. "
            "Read each training setup and pick the choice that achieves the best "
            f"**{selection_metric}** after the stated training budget. Give the answer "
            "as a single letter wrapped in <answer></answer> tags."
        )

    dataset_nl = _read_template(f"dataset/{q['family']}.md")
    if not dataset_nl:
        dataset_nl = (
            "Univariate regression on [0, 1]. Input and target are 1-D scalars. "
            f"Train size: {params['train_size']}, test size: {params['test_size']}."
        )

    is_classification = q["family"] in TABULAR_CLASSIFICATION_FAMILIES
    total_samples_seen = _question_total_samples_seen(q["budget"])
    single_axis = q["type"] in SINGLE_AXIS_TYPES and not (
        isinstance(q["budget"], dict) and q["budget"].get("mixed")
    )

    budget_heading = (
        "## Sample budget"
        if total_samples_seen is None
        else "## Sample budget (same for all choices)"
    )
    parts = [
        header.strip(),
        "",
        "## Dataset",
        dataset_nl.strip(),
        "",
    ]
    if is_classification:
        parts.extend(
            [
                "### Data-generating and classification rule",
                format_synthetic_tabular_classification_rule(params),
                "",
            ]
        )
    else:
        synth_source = (dataset_path / "synthesize.py").read_text(encoding="utf-8")
        synth_code = excerpt_synthesize_py(synth_source)
        parts.extend(["### Synthesis (PyTorch)", "```python", synth_code, "```", ""])
    parts.extend(
        [
            "### Data splits and training protocol",
            format_dataset_protocol(params, family=q["family"], device=str(eval_meta.get("device", "cpu"))),
            "",
            budget_heading,
        ]
    )
    if single_axis and q["choices"]:
        first_cand = read_json(
            artifact_root / q["choices"][0]["candidate_path"] / "candidate_spec.json"
        )
        parts.append(format_training_schedule(first_cand["budget"]))
    elif total_samples_seen is not None:
        parts.extend(
            [
                f"- total_samples_seen: {total_samples_seen}",
                "",
                "Each choice specifies its own `training_steps` and `batch_size` below; "
                "they must satisfy `training_steps × batch_size = total_samples_seen`.",
            ]
        )
    else:
        parts.extend(
            [
                "- budgets differ across choices",
                "",
                "Each choice specifies its own training budget below.",
            ]
        )
    protocol = format_ranking_protocol(
        n_seeds=int(eval_meta["n_seeds"]),
        base_seed=int(eval_meta["base_seed"]),
        selection_metric=selection_metric,
        device=str(eval_meta.get("device", "cpu")),
    )
    if task == "ranking":
        # same ground-truth protocol, but the closing line must state the order's
        # direction instead of naming a single winner
        protocol = "\n".join(
            protocol.splitlines()[:-1] + [format_ranking_objective(selection_metric)]
        )
    parts.extend(["", "## Evaluation metric", protocol, ""])

    ranking_loss_only = task == "ranking" and q.get("type") == "loss_only"
    if ranking_loss_only:
        parts.extend(_ranking_shared_sections(q, artifact_root, question_path))

    parts.append("## Choices")

    for choice in q["choices"]:
        cand_path = artifact_root / choice["candidate_path"]
        cand_spec = read_json(cand_path / "candidate_spec.json")
        _sync_candidate_files(cand_path, cand_spec)

        parts.extend(
            [
                "",
                f"### Choice {choice['letter']}",
                "",
            ]
        )
        if ranking_loss_only:
            loss_code = excerpt_loss_py((cand_path / "loss.py").read_text(encoding="utf-8"))
            parts.extend(
                [
                    "**Loss**",
                    format_loss_nl(cand_spec["loss"]),
                    "",
                    "```python",
                    loss_code,
                    "```",
                ]
            )
            continue
        model_code = excerpt_model_py((cand_path / "model.py").read_text(encoding="utf-8"))
        loss_code = excerpt_loss_py((cand_path / "loss.py").read_text(encoding="utf-8"))
        opt_code = excerpt_optimizer_py((cand_path / "optimizer.py").read_text(encoding="utf-8"))

        if not single_axis:
            parts.extend(
                [
                    "**Training schedule**",
                    format_training_schedule(cand_spec["budget"]),
                    "",
                ]
            )
        parts.extend(
            [
                "**Model (natural language)**",
                format_model_nl(cand_spec["model"]),
                "",
                "**Model code**",
                "```python",
                model_code,
                "```",
                "",
                "**Optimizer**",
                format_optimizer_nl(cand_spec["optimizer"]),
                "",
                "```python",
                opt_code,
                "```",
                "",
                "**Loss**",
                format_loss_nl(cand_spec["loss"]),
                "",
                "```python",
                loss_code,
                "```",
            ]
        )

    letters = [c["letter"] for c in q["choices"]]
    if task == "ranking":
        parts.extend(
            [
                "",
                format_ranking_answer_section(letters, answer_order=_gold_order(q)),
            ]
        )
        return "\n".join(parts)
    parts.extend(["", format_mcq_answer_section(letters)])
    return "\n".join(parts)


def write_prompt(
    question_path: Path,
    *,
    dataset_path: Path | None = None,
    artifact_root: Path | None = None,
    task: str = "mcq",
) -> Path:
    text = render_prompt(
        question_path,
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        task=task,
    )
    out = question_path / "prompt.txt"
    out.write_text(text, encoding="utf-8")
    return out


# The preamble sentence older revisers wrote to pre-announce the two-field answer
# format. It repeats the answer section, and it names a field the contract no longer
# asks for, so normalising a stored prompt drops it.
_STALE_FORMAT_PARA = re.compile(
    r"\n*Answer format is specified in the final section.*?tags\.\n*", re.DOTALL
)
_LETTERS_IN_SECTION = re.compile(r"Choose exactly one of ([A-Z](?:, [A-Z])*)\.\s")


def canonicalize_mcq_prompt(prompt: str, letters: list[str] | None = None) -> str:
    """Rewrite the answer section of a stored pick-one prompt onto the current contract.

    Prompt assembly lives in this module, so bringing an older prompt up to date belongs
    here too. The alternative -- each tool growing its own copy of the answer section --
    is how the pool builders and the merge tool would drift apart, and a prompt that
    still asks for `<explanation>` is rejected outright by the parquet converter.

    Everything above the answer section is returned byte-identical, so a prompt already
    on the current contract round-trips unchanged.
    """
    cut = prompt.rfind("## Your answer")
    if cut < 0:
        raise ValueError("no '## Your answer' section")
    head = _STALE_FORMAT_PARA.sub("", prompt[:cut]).rstrip()
    if letters is None:
        found = _LETTERS_IN_SECTION.search(prompt, cut)
        letters = found.group(1).split(", ") if found else ["A", "B", "C"]
    return head + "\n\n" + format_mcq_answer_section(letters)

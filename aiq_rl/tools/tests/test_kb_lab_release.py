import contextlib
import copy
import hashlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kb_lab


class LabReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        self.lab = self.root / "release_lab"
        self.pool = self.root / "questions.jsonl"
        self.split = self.root / "split.json"
        self.mock = patch.object(kb_lab, "DATASETS", self.datasets)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def dataset(self, family, dataset_id, seed=42):
        directory = self.datasets / family / dataset_id
        directory.mkdir(parents=True)
        (directory / "dataset_spec.json").write_text(json.dumps({"family": family, "dataset_id": dataset_id,
                    "params": {"seed": seed}, "selection_metric": "loss"}))
        for i in range(2):
            candidate = directory / "candidates/set_1" / f"c_{i}"
            (candidate / "results").mkdir(parents=True)
            (candidate / "candidate_spec.json").write_text(json.dumps({"budget": {"training_steps": 1, "batch_size": 1},
                      "model": {"type": "mlp", "width": 4, "depth": 1}, "optimizer": {"type": "sgd"}, "loss": {}}))
            (candidate / "results/summary.json").write_text(json.dumps({"selection_metric": "loss", "mean_loss": i + 1.}))

    def write_pool(self, questions):
        self.pool.write_text("".join(json.dumps(q) + "\n" for q in questions))

    def question(self, qid, source, family, group, meta=None, prompt="measured fixture"):
        return {"question_id": qid, "source": source, "family": family, "group": group,
                "messages": [{"role": "user", "content": prompt}], "meta": meta or {}}

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            kb_lab.build(SimpleNamespace(pool=self.pool, split=[str(self.split)], lab_dir=self.lab))

    def test_release_split_excludes_all_four_sources_and_canonical_members(self):
        sources = ("arch170", "ranking_v2", "ranking_v3", "dataflip500")
        questions = []
        mapping = {}
        for i, source in enumerate(sources):
            dataset_id, family = f"held{i}", f"family{i}"
            self.dataset(family, dataset_id)
            self.dataset(family, "safe")
            meta = ({"candidate_set": f"datasets/{family}/{dataset_id}/candidates/set_1"} if i % 2 == 0
                    else {"dataset_id": dataset_id})
            q = self.question(f"q{i}", source, family, f"original_{i}", meta)
            questions.append(q)
            mapping[q["question_id"]] = f"Gcanonical{i}"
        # Another source member in the same canonical held-out group must also stay outside the lab.
        self.dataset("extra", "same_group")
        questions.append(self.question("linked", "ranking_v3", "extra", "original_link",
                                       {"dataset_path": "datasets/extra/same_group"}))
        mapping["linked"] = mapping["q0"]
        self.write_pool(questions)
        self.split.write_text(json.dumps({"train": [], "val": ["q0", "q1"], "test": ["q2", "q3"],
                                          "group_by_question": mapping}))
        self.build()
        allowed = set((self.lab / "allowed_datasets.txt").read_text().split())
        self.assertEqual(allowed, {f"family{i}/safe" for i in range(4)})
        manifest = json.loads((self.lab / "lab_manifest.json").read_text())
        self.assertEqual(manifest["pool"]["sha256"], hashlib.sha256(self.pool.read_bytes()).hexdigest())
        self.assertEqual(manifest["splits"][0]["sha256"], hashlib.sha256(self.split.read_bytes()).hexdigest())
        self.assertEqual(len(manifest["excluded_dataset_dirs"]), 5)

    def test_runtime_split_with_canonical_groups_uses_explicit_ids_and_prompt_seeds(self):
        self.dataset("seed_family", "seed_held", 876543210)
        self.dataset("seed_family", "safe", 123)
        self.dataset("path_family", "path_held", 77)
        questions = [self.question("seed", "dataflip500", "seed_family", "old_seed_group",
                                   prompt="synthesis seed=876543210"),
                     self.question("path", "ranking_v2", "path_family", "old_path_group",
                                   {"dataset": "datasets/path_family/path_held"})]
        self.write_pool(questions)
        self.split.write_text(json.dumps({"stream": {}, "eval": ["seed"], "test": ["path"],
                                          "eval_groups": ["Gcanonical_seed", "Gcanonical_path"]}))
        self.build()
        self.assertEqual(set((self.lab / "allowed_datasets.txt").read_text().split()), {"seed_family/safe"})

    def test_legacy_runtime_group_paths_still_exclude(self):
        questions = [self.question("q", "arch170", "family", "datasets/family/held")]
        self.write_pool(questions)
        self.split.write_text(json.dumps({"eval_groups": ["datasets/family/held"]}))
        dirs, seeds = kb_lab.held_out([self.split], self.pool)
        self.assertEqual(dirs, {"family/held"})
        self.assertEqual(seeds, set())

    def test_explicit_short_and_nested_seed_matches_dataset(self):
        self.dataset("family", "held_meta", 17)
        self.dataset("family", "held_prompt", 23)
        self.dataset("family", "safe", 91)
        self.write_pool([self.question("meta", "ranking_v3", "family", "g1",
                                      {"params": {"generation": {"data_seed": 17}}}),
                         self.question("prompt", "dataflip500", "family", "g2", prompt="generated with seed=23")])
        self.split.write_text(json.dumps({"val": ["meta"], "test": ["prompt"]}))
        self.build()
        self.assertEqual(set((self.lab / "allowed_datasets.txt").read_text().split()), {"family/safe"})

    def test_runtime_eval_group_uses_adjacent_release_group_map(self):
        self.write_pool([self.question("q1", "arch170", "family", "original1",
                                      {"candidate_set": "datasets/family/held/candidates/set_1"}),
                         self.question("q2", "ranking_v2", "extra", "original2",
                                      {"dataset_id": "also_held"})])
        self.split.write_text(json.dumps({"train": [], "val": ["q1", "q2"], "test": [],
                                          "group_by_question": {"q1": "Gshared", "q2": "Gshared"}}))
        runtime = self.root / "runtime.json"
        runtime.write_text(json.dumps({"stream": {}, "eval": ["q1"], "test": [], "eval_groups": ["Gshared"]}))
        dirs, _ = kb_lab.held_out([runtime], self.pool)
        self.assertEqual(dirs, {"family/held", "extra/also_held"})

    def test_required_allowlist_refuses_missing_or_tampered_inputs_before_training_import(self):
        self.dataset("family", "safe")
        self.dataset("family", "held")
        self.write_pool([self.question("q", "arch170", "family", "datasets/family/held")])
        self.split.write_text(json.dumps({"eval": ["q"], "test": []}))
        job = {"dataset": "family/safe", "candidates": []}
        with self.assertRaisesRegex(ValueError, "allowlist or manifest is missing"):
            kb_lab.run_job(job, lab_dir=self.lab, require_allowlist=True, pool=self.pool)
        self.build()
        with self.assertRaisesRegex(ValueError, "not in the lab"):
            kb_lab.run_job({**job, "dataset": "family/held"}, lab_dir=self.lab, require_allowlist=True, pool=self.pool)
        with (self.pool).open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "pool hash mismatch"):
            kb_lab.run_job(job, lab_dir=self.lab, require_allowlist=True, pool=self.pool)
        with (self.lab / "allowed_datasets.txt").open("a") as f:
            f.write("family/held\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            kb_lab.run_job(job, lab_dir=self.lab, require_allowlist=True)

    def test_unknown_heldout_id_and_path_traversal_fail(self):
        self.write_pool([self.question("q", "arch170", "family", "g")])
        self.split.write_text(json.dumps({"val": ["missing"], "test": []}))
        with self.assertRaisesRegex(ValueError, "absent from pool"):
            self.build()
        self.assertFalse(self.lab.exists())
        with self.assertRaisesRegex(ValueError, "dataset must"):
            kb_lab.run_job({"dataset": "../elsewhere", "candidates": []}, lab_dir=self.lab)

    def test_changed_split_refuses_old_allowlist(self):
        self.dataset("family", "safe")
        self.write_pool([self.question("q", "arch170", "family", "g")])
        self.split.write_text(json.dumps({"val": [], "test": []}))
        self.build()
        self.split.write_text(json.dumps({"val": ["q"], "test": []}))
        with self.assertRaisesRegex(ValueError, "split hash mismatch"):
            kb_lab.run_job({"dataset": "family/safe", "candidates": []}, lab_dir=self.lab,
                           require_allowlist=True, pool=self.pool)

    def test_validation_api_returns_allowlist_and_rejects_changed_pool_and_split(self):
        self.dataset("family", "safe")
        self.write_pool([self.question("q", "arch170", "family", "g")])
        self.split.write_text(json.dumps({"val": [], "test": []}))
        self.build()
        self.assertEqual(kb_lab.validate_allowlist(self.lab, self.pool), {"family/safe"})
        different_pool = self.root / "other.jsonl"
        different_pool.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "pool hash mismatch"):
            kb_lab.validate_allowlist(self.lab, different_pool)
        self.split.write_text(json.dumps({"val": ["q"], "test": []}))
        with self.assertRaisesRegex(ValueError, "split hash mismatch"):
            kb_lab.validate_allowlist(self.lab, self.pool)

    def measured_fixture(self, summary, cached, curves=None, error=None, capture_sources=True, mutate_source=False):
        self.dataset("family", "safe")
        job = {"dataset": "family/safe", "candidates": [{
            "model": {"type": "mlp", "width": 4, "depth": 1},
            "optimizer": {"type": "SGD", "lr": .001},
            "loss": {"loss_id": "mse"}, "budget": {"training_steps": 1, "batch_size": 1}}]}
        if capture_sources:
            job["capture_executable_sources"] = True
        directories = []

        def save_outputs(directory):
            (directory / "results").mkdir(exist_ok=True)
            (directory / "results/summary.json").write_text(json.dumps(summary))
            if curves is not None:
                (directory / "results/curves.npz").write_bytes(curves)

        def write_candidate(spec, directory, model_type):
            directories.append(directory)
            for name in kb_lab.EXECUTABLE_FILES:
                (directory / name).write_text(f"EXECUTED_SOURCE = {name!r}\n")
            if cached and directory == directories[0]:
                save_outputs(directory)

        def fixture_ground_truth(directory, profile, dataset, *, sync_files=True):
            self.assertEqual(sync_files, not capture_sources)
            if error is not None:
                raise error
            save_outputs(directory)
            if curves is None:
                (directory / "results/curves.npz").write_bytes(b"fixture curves")
            if mutate_source:
                (directory / "train.py").write_text("changed during execution\n")
            return summary

        ground_truth = Mock(side_effect=fixture_ground_truth)
        modules = {"architecture_iq": SimpleNamespace(),
                   "architecture_iq.candidates": SimpleNamespace(),
                   "architecture_iq.candidates.generator": SimpleNamespace(write_candidate=write_candidate),
                   "architecture_iq.registry": SimpleNamespace(ensure_registries=lambda: None,
                                                               get_model_type=lambda name: name),
                   "architecture_iq.ground_truth": SimpleNamespace(),
                   "architecture_iq.ground_truth.runner": SimpleNamespace(run_ground_truth=ground_truth),
                   "architecture_iq.profile": SimpleNamespace(load_profile=lambda name: {"fixture": True})}
        with patch.dict(sys.modules, modules), patch.dict(kb_lab.os.environ), \
                patch.object(sys, "path", list(sys.path)), \
                patch.object(kb_lab.cf, "ProcessPoolExecutor", ThreadPoolExecutor):
            result = kb_lab.run_job(job, workers=1, lab_dir=self.lab)["results"][0]
        self.assertEqual(result["candidate"], job["candidates"][0])
        return result, directories[0], ground_truth

    def test_cached_measurement_preserves_failed_seed_and_curve_provenance(self):
        seeds = [{"seed": 37, "failed": False, "final_test_mse": .2},
                 {"seed": 38, "failed": True, "final_test_mse": None}]
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2, "std_test_mse": 0.,
                   "failed_seeds": 1, "excluded": True, "n_seeds": 2, "base_seed": 37,
                   "seed_results": seeds}
        result, directory, ground_truth = self.measured_fixture(summary, cached=True, curves=b"fixture curves")
        ground_truth.assert_not_called()
        self.assertTrue(result["cached"])
        self.assertEqual(result["source_provenance"]["status"], "unverified")
        self.assertEqual(result["source_provenance"]["omitted"], list(kb_lab.EXECUTABLE_FILES))
        for key in ("failed_seeds", "excluded", "n_seeds", "base_seed", "seed_results"):
            self.assertEqual(result[key], summary[key])
        self.assertEqual(result["mean"], .2)
        self.assertEqual(result["std"], 0.)
        self.assertEqual(set(result["measurement_files"]),
                         {"candidate_spec.json", "results/summary.json", "results/curves.npz"})
        for name, metadata in result["measurement_files"].items():
            payload = (directory / name).read_bytes()
            self.assertEqual(metadata, {"path": str((directory / name).resolve()),
                                       "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})

    def test_cached_old_summary_retains_unknown_seed_fields(self):
        result, directory, ground_truth = self.measured_fixture(
            {"selection_metric": "loss", "mean_loss": .3}, cached=True)
        ground_truth.assert_not_called()
        self.assertTrue(result["cached"])
        for key in ("std", "failed_seeds", "excluded", "n_seeds", "base_seed", "seed_results"):
            self.assertIsNone(result[key])
        self.assertEqual(set(result["measurement_files"]), {"candidate_spec.json", "results/summary.json"})
        self.assertFalse((directory / "results/curves.npz").exists())

    def test_fresh_measurement_preserves_runner_seed_results_exactly(self):
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2, "std_test_mse": .01,
                   "failed_seeds": 0, "excluded": False, "n_seeds": 2, "base_seed": 17,
                   "seed_results": [{"seed": 17, "failed": False, "final_test_mse": .19},
                                    {"failed": False, "final_test_mse": .21}]}
        result, directory, ground_truth = self.measured_fixture(summary, cached=False, curves=b"fresh curves")
        ground_truth.assert_called_once()
        executed = ground_truth.call_args.args[0]
        self.assertEqual(executed.parent, directory.resolve() / "results")
        self.assertNotEqual(executed, directory)
        self.assertEqual(ground_truth.call_args.args[1:], ({"fixture": True}, self.datasets / "family/safe"))
        self.assertEqual(ground_truth.call_args.kwargs, {"sync_files": False})
        self.assertFalse(result["cached"])
        self.assertEqual(result["seed_results"], summary["seed_results"])
        self.assertNotIn("seed", result["seed_results"][1])
        for key in ("failed_seeds", "excluded", "n_seeds", "base_seed"):
            self.assertEqual(result[key], summary[key])
        self.assertEqual(result["measurement_files"]["results/curves.npz"]["sha256"],
                         hashlib.sha256(b"fresh curves").hexdigest())
        self.assertEqual(json.loads((directory / "results/summary.json").read_text()), summary)
        self.assertEqual(result["source_provenance"]["status"], "verified")
        self.assertEqual(result["source_provenance"]["omitted"], [])
        for name in kb_lab.EXECUTABLE_FILES:
            metadata = result["measurement_files"][f"executed/{name}"]
            source = Path(metadata["path"])
            self.assertEqual(source.parent, executed)
            self.assertEqual(source.read_bytes(), (directory / name).read_bytes())
            self.assertEqual(kb_lab._file_evidence(source), {key: metadata[key] for key in ("sha256", "bytes")})
        (directory / "model.py").write_text("CURRENT_SOURCE = 'changed after execution'\n")
        cached = kb_lab._measurement_result(str(directory), summary, cached=True, capture_sources=True)
        self.assertEqual(cached["source_provenance"]["status"], "verified")
        self.assertEqual(cached["measurement_files"]["executed/model.py"],
                         result["measurement_files"]["executed/model.py"])

    def test_corrupt_execution_source_or_measurement_downgrades_provenance(self):
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2}
        result, directory, ground_truth = self.measured_fixture(summary, cached=False, curves=b"fixture")
        source = Path(result["measurement_files"]["executed/train.py"]["path"])
        original = source.read_bytes()
        source.write_text("changed source\n")
        corrupted = kb_lab._measurement_result(str(directory), summary, cached=True, capture_sources=True)
        self.assertEqual(corrupted["source_provenance"]["status"], "unverified")
        self.assertIn("executed source hash mismatch", corrupted["source_provenance"]["reason"])
        self.assertFalse(any(name.startswith("executed/") for name in corrupted["measurement_files"]))
        source.write_bytes(original)
        (directory / "results/curves.npz").write_bytes(b"changed measurement")
        corrupted = kb_lab._measurement_result(str(directory), summary, cached=True, capture_sources=True)
        self.assertEqual(corrupted["source_provenance"]["status"], "unverified")
        self.assertIn("measurement hash mismatch", corrupted["source_provenance"]["reason"])
        ground_truth.assert_called_once()

    def test_execution_source_mutation_does_not_publish_measurement(self):
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2}
        result, directory, ground_truth = self.measured_fixture(summary, cached=False,
            curves=b"fixture", mutate_source=True)
        ground_truth.assert_called_once()
        self.assertIn("executed source changed during measurement: train.py", result["error"])
        self.assertEqual(result["source_provenance"]["status"], "unverified")
        self.assertFalse((directory / "results/summary.json").exists())
        self.assertFalse((directory / kb_lab.EXECUTION_MANIFEST).exists())

    def test_executable_snapshot_is_loaded_and_manifest_binds_its_outputs(self):
        directory = self.lab / "experiments/family/safe/x_fixture"
        directory.mkdir(parents=True)
        (directory / "candidate_spec.json").write_text(json.dumps({"model": {"type": "fixture"}}))
        for name in kb_lab.EXECUTABLE_FILES:
            (directory / name).write_text("raise RuntimeError('stale root source must not execute')\n")
        loader_path = kb_lab.BENCH / "src/architecture_iq/runtime/loader.py"
        loader_spec = importlib.util.spec_from_file_location("fixture_runtime_loader", loader_path)
        loader = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(loader)
        sources = {"model.py": "TOKEN = 'model'\n",
                   "optimizer.py": "TOKEN = 'optimizer'\n",
                   "loss.py": "TOKEN = 'loss'\n",
                   "train.py": "import model, optimizer, loss\ndef read_sources():\n    return model.TOKEN, optimizer.TOKEN, loss.TOKEN\n"}
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2,
                   "seed_results": [{"seed": 7, "failed": False, "final_test_mse": .2}]}
        executed = []

        def write_candidate(spec, snapshot, model_type):
            for name, text in sources.items():
                (snapshot / name).write_text(text)

        def ground_truth(snapshot, profile, dataset, *, sync_files):
            self.assertFalse(sync_files)
            loaded = loader.load_candidate_train(snapshot)
            self.assertEqual(loaded.read_sources(), ("model", "optimizer", "loss"))
            executed.append(snapshot)
            (snapshot / "results").mkdir()
            (snapshot / "results/summary.json").write_text(json.dumps(summary))
            (snapshot / "results/curves.npz").write_bytes(b"loaded-source fixture curve")
            return summary

        modules = {"architecture_iq": SimpleNamespace(),
                   "architecture_iq.candidates": SimpleNamespace(),
                   "architecture_iq.candidates.generator": SimpleNamespace(write_candidate=write_candidate),
                   "architecture_iq.registry": SimpleNamespace(ensure_registries=lambda: None, get_model_type=lambda name: name),
                   "architecture_iq.ground_truth": SimpleNamespace(),
                   "architecture_iq.ground_truth.runner": SimpleNamespace(run_ground_truth=ground_truth),
                   "architecture_iq.profile": SimpleNamespace(load_profile=lambda name: {"fixture": True})}
        with patch.dict(sys.modules, modules), patch.dict(kb_lab.os.environ), patch.object(sys, "path", list(sys.path)):
            result = kb_lab._gt_worker((str(directory), str(self.datasets / "family/safe"), True))
        self.assertEqual(result["source_provenance"]["status"], "verified")
        manifest = json.loads((directory / kb_lab.EXECUTION_MANIFEST).read_text())
        self.assertEqual(directory.resolve() / manifest["execution_dir"], executed[0])
        for name, text in sources.items():
            metadata = result["measurement_files"][f"executed/{name}"]
            self.assertEqual(Path(metadata["path"]).read_text(), text)
            self.assertEqual(metadata["sha256"], hashlib.sha256(text.encode()).hexdigest())
        for name in kb_lab.RESULT_FILES:
            self.assertEqual(manifest["measurements"][name], kb_lab._file_evidence(directory / name))
        self.assertEqual(result["seed_results"], summary["seed_results"])

    def test_old_job_keeps_original_runner_path_and_capture_allowlist(self):
        summary = {"selection_metric": "test_mse", "mean_test_mse": .2,
                   "seed_results": [{"seed": 1, "failed": False, "final_test_mse": .2}]}
        result, directory, ground_truth = self.measured_fixture(summary, cached=False,
            curves=b"legacy fixture", capture_sources=False)
        ground_truth.assert_called_once_with(directory, {"fixture": True}, self.datasets / "family/safe")
        self.assertEqual(set(result["measurement_files"]), set(kb_lab.RESULT_FILES))
        self.assertNotIn("source_provenance", result)
        self.assertFalse((directory / kb_lab.EXECUTION_MANIFEST).exists())
        self.assertFalse(list((directory / "results").glob("execution_*")))
        for filename in result["measurement_files"]:
            if filename not in ("candidate_spec.json", "results/summary.json", "results/curves.npz"):
                self.fail(f"old research capture would reject {filename}")
        result = kb_lab._measurement_result(str(directory), summary, cached=True)
        self.assertEqual(set(result["measurement_files"]), set(kb_lab.RESULT_FILES))
        self.assertNotIn("source_provenance", result)

    def test_canonical_snapshot_preserves_tiny_cpu_fixture_measurement(self):
        sys.path.insert(0, str(kb_lab.BENCH / "src"))
        import torch
        from architecture_iq.candidates.generator import write_candidate
        from architecture_iq.profile import load_profile
        from architecture_iq.registry import ensure_registries, get_model_type
        from architecture_iq.ground_truth.runner import run_ground_truth
        import architecture_iq.profile as profile_module
        import numpy as np

        ensure_registries()
        dataset = self.root / "synthetic_cpu_fixture"
        dataset.mkdir()
        x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = torch.tensor([0, 1, 1, 0])
        torch.save({"x": x, "y": y}, dataset / "train.pt")
        torch.save({"x": x, "y": y}, dataset / "test.pt")
        (dataset / "dataset_spec.json").write_text(json.dumps({"family": "synthetic_tabular_classification",
            "selection_metric": "test_ce", "params": {}, "significance": {}}))
        spec = {"schema_version": "1.0", "candidate_id": "x_tiny_fixture", "dataset_id": "fixture",
            "family": "synthetic_tabular_classification", "budget": {"training_steps": 3, "batch_size": 2, "total_samples_seen": 6},
            "model": {"type": "mlp", "input_dim": 2, "output_dim": 2, "depth": 1, "width": 4,
                      "activation": "leaky_relu", "layer_norm": [False], "residual": False},
            "optimizer": {"type": "Adam", "lr": .001, "weight_decay": .01, "betas": [.9, .999]},
            "loss": {"loss_id": "cross_entropy"}, "execution": {"device": "cpu"}}
        legacy, capture = self.root / "legacy_fixture", self.root / "capture_fixture"
        for directory in (legacy, capture):
            write_candidate(spec, directory, get_model_type("mlp"))
        profile = load_profile("v1.5")
        profile.ground_truth["n_seeds"] = 1
        profile.ground_truth["base_seed"] = 7
        with patch.dict(kb_lab.os.environ, {"ARCHITECTURE_IQ_SEED_WORKERS": "1", "ARCHITECTURE_IQ_TORCH_THREADS": "1"}), \
             patch.object(profile_module, "load_profile", return_value=profile):
            baseline = run_ground_truth(legacy, profile, dataset)
            captured = kb_lab._gt_worker((str(capture), str(dataset), True))
        self.assertEqual(captured["source_provenance"]["status"], "verified")
        saved = json.loads((capture / "results/summary.json").read_text())
        self.assertEqual(saved, baseline)
        self.assertEqual(captured["seed_results"], baseline["seed_results"])
        with np.load(legacy / "results/curves.npz") as before, np.load(capture / "results/curves.npz") as after:
            self.assertEqual(before.files, after.files)
            for name in before.files:
                np.testing.assert_array_equal(before[name], after[name])
        for name in kb_lab.EXECUTABLE_FILES:
            metadata = captured["measurement_files"][f"executed/{name}"]
            self.assertEqual(Path(metadata["path"]).read_bytes(), (legacy / name).read_bytes())

    def test_process_runner_lab_cache_and_repo_roundtrip(self):
        sys.path.insert(0, str(kb_lab.BENCH / "src"))
        import torch
        import numpy as np
        from architecture_iq.candidates.generator import write_candidate, validate_process_request
        from architecture_iq.profile import load_profile
        from architecture_iq.registry import ensure_registries, get_model_type
        from architecture_iq.ground_truth.runner import run_ground_truth
        from architecture_iq.runtime.loader import load_candidate_train
        import architecture_iq.profile as profile_module

        ensure_registries()
        dataset = self.datasets / "synthetic_tabular_classification" / "fixture"
        dataset.mkdir(parents=True)
        x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = torch.tensor([0, 1, 1, 0])
        torch.save({"x": x, "y": y}, dataset / "train.pt")
        torch.save({"x": x, "y": y}, dataset / "test.pt")
        (dataset / "dataset_spec.json").write_text(json.dumps({"family": "synthetic_tabular_classification",
            "dataset_id": "fixture", "selection_metric": "test_ce", "params": {}, "significance": {}}))
        candidate = {"model": {"type": "mlp", "input_dim": 2, "output_dim": 2, "depth": 1, "width": 4,
                              "activation": "leaky_relu", "layer_norm": [False], "residual": False},
            "optimizer": {"type": "Adam", "lr": .001, "weight_decay": .01, "betas": [.9, .999]},
            "loss": {"loss_id": "cross_entropy"}, "budget": {"training_steps": 3, "batch_size": 2}}
        spec = {**copy.deepcopy(candidate), "schema_version": "1.0", "candidate_id": "x_baseline_fixture",
                "dataset_id": "fixture", "family": "synthetic_tabular_classification", "execution": {"device": "cpu"}}
        spec["budget"]["total_samples_seen"] = 6
        legacy = self.root / "baseline"
        write_candidate(spec, legacy, get_model_type("mlp"))
        profile = load_profile("v1.5")
        profile.ground_truth.update(n_seeds=1, base_seed=7)
        job = {"dataset": "synthetic_tabular_classification/fixture", "candidates": [candidate],
               "capture_executable_sources": True, "process": {"steps": [3, 1], "max_eval_samples": 4}}
        with patch.dict(kb_lab.os.environ, {"ARCHITECTURE_IQ_SEED_WORKERS": "1", "ARCHITECTURE_IQ_TORCH_THREADS": "1"}), \
             patch.object(profile_module, "load_profile", return_value=profile), \
             patch.object(kb_lab.cf, "ProcessPoolExecutor", ThreadPoolExecutor):
            baseline = run_ground_truth(legacy, profile, dataset)
            captured = kb_lab.run_job(job, workers=1, lab_dir=self.lab)["results"][0]
            cached = kb_lab.run_job(job, workers=1, lab_dir=self.lab)["results"][0]
        self.assertNotIn("error", captured)
        self.assertFalse(captured["cached"])
        self.assertTrue(cached["cached"])
        self.assertEqual(captured["process_provenance"]["status"], "verified")
        self.assertEqual(captured["seed_results"], baseline["seed_results"])
        summary_path = Path(captured["measurement_files"]["results/summary.json"]["path"])
        directory = summary_path.parent.parent
        recorded_summary = json.loads(summary_path.read_text())
        ordinary = {key: value for key, value in recorded_summary.items() if key != "process"}
        self.assertEqual({key: value for key, value in ordinary.items() if key != "candidate_id"},
                         {key: value for key, value in baseline.items() if key != "candidate_id"})
        with np.load(legacy / "results/curves.npz") as before, np.load(directory / "results/curves.npz") as after:
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key])
        process_json = directory / "results/process/seed_7.json"
        record = json.loads(process_json.read_text())
        self.assertEqual([row["step"] for row in record["steps"]], [1, 3])
        self.assertEqual(record["seed"], 7)
        self.assertEqual(record["profile"], "v1.5")
        self.assertEqual(record["optimizer_defaults"]["eps"], 1e-8)
        self.assertEqual(record["inputs"]["test_x"]["sha256"], hashlib.sha256(x.numpy().tobytes()).hexdigest())
        with np.load(directory / "results/process/seed_7.npz") as arrays:
            logits, targets = torch.from_numpy(arrays["logits"]), torch.from_numpy(arrays["targets"])
            self.assertEqual(float(torch.nn.functional.cross_entropy(logits, targets)), baseline["mean_test_ce"])
            np.testing.assert_array_equal(arrays["errors"], arrays["predictions"] != arrays["targets"])
            other = logits.clone()
            other.scatter_(1, targets[:, None], float("-inf"))
            expected = logits.gather(1, targets[:, None]).reshape(-1) - other.max(1).values
            np.testing.assert_array_equal(arrays["true_class_margins"], expected.numpy())
        recorder_spec = json.loads((directory / "candidate_spec.json").read_text())
        instrumented = load_candidate_train(Path(captured["measurement_files"]["executed/train.py"]["path"]).parent)
        plain = load_candidate_train(legacy)
        def state_after(module, process=None):
            made = {}
            sampled = []
            factory = module.build_optimizer
            randint = torch.randint
            def remember(model):
                made["model"], made["optimizer"] = model, factory(model)
                return made["optimizer"]
            def sample(*args, **kwargs):
                indices = randint(*args, **kwargs)
                sampled.append(indices.clone())
                return indices
            with patch.object(module, "build_optimizer", side_effect=remember), patch.object(torch, "randint", side_effect=sample):
                kwargs = {"steps": 3, "batch_size": 2, "seed": 7}
                if process is not None:
                    kwargs["process"] = process
                result = module.train_and_eval(x, y, x, y, **kwargs)
            made["samples"] = sampled
            return made, result
        off, _ = state_after(plain)
        on, _ = state_after(instrumented, recorder_spec["process"])
        for left, right in zip(off["samples"], on["samples"]):
            self.assertTrue(torch.equal(left, right))
        for observation in record["steps"]:
            indices = on["samples"][observation["step"] - 1]
            self.assertEqual(observation["minibatch"]["sha256"], hashlib.sha256(indices.numpy().tobytes()).hexdigest())
        self.assertEqual(record["minibatch_stream_sha256"],
                         hashlib.sha256(b"".join(indices.numpy().tobytes() for indices in on["samples"])).hexdigest())
        for left, right in zip(off["model"].parameters(), on["model"].parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(off["optimizer"].state.values(), on["optimizer"].state.values()):
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]))
        manifest = json.loads((directory / kb_lab.EXECUTION_MANIFEST).read_text())
        self.assertIn("results/process/seed_7.json", manifest["measurements"])
        original = process_json.read_bytes()
        process_json.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            kb_lab._measurement_result(str(directory), recorded_summary, True, True)
        process_json.write_bytes(original + b" " )
        with self.assertRaisesRegex(ValueError, "diagnostic hash mismatch"):
            kb_lab._measurement_result(str(directory), recorded_summary, True, True)
        process_json.write_bytes(original)
        missing = copy.deepcopy(recorded_summary)
        missing.pop("process")
        with self.assertRaisesRegex(ValueError, "missing or mismatched"):
            kb_lab._measurement_result(str(directory), missing, True, True)
        plain_job = dict(job)
        plain_job.pop("process")
        with patch.object(profile_module, "load_profile", return_value=profile), \
             patch.object(kb_lab.cf, "ProcessPoolExecutor", ThreadPoolExecutor):
            plain_result = kb_lab.run_job(plain_job, workers=1, lab_dir=self.lab)["results"][0]
        self.assertNotEqual(plain_result["measurement_files"]["candidate_spec.json"]["path"],
                            captured["measurement_files"]["candidate_spec.json"]["path"])
        normalized = validate_process_request(job["process"], recorder_spec)
        self.assertEqual(normalized, recorder_spec["process"])
        import kb_science_loop as loop
        import threading
        obj = loop.Loop.__new__(loop.Loop)
        obj.run = self.root / "research_roundtrip"
        obj.run.mkdir()
        obj.lock = threading.RLock()
        obj.worker_path = obj.run / "jobs/Jfixture/job.json"
        obj.worker_path.parent.mkdir(parents=True)
        obj.worker_job = {"id": "Jfixture", "origin_epoch": 3}
        obj.cur_epoch = 3
        obj.lab_sem = threading.Semaphore(1)
        obj.args = SimpleNamespace(research_max_cands=2, lab=str(self.lab / "lab_db.jsonl"),
                                   lab_workers=1, pool=str(self.root / "fixture_pool.jsonl"))
        base = {**candidate, "family": "synthetic_tabular_classification", "dataset_id": "fixture",
                "dataset": {}, "metric": "test_ce"}
        response = SimpleNamespace(returncode=0, stdout=json.dumps({"dataset": job["dataset"], "results": [captured]}))
        with patch.object(obj, "lab_row", return_value=base), \
             patch.object(obj, "lab_dataset_params", return_value={}), \
             patch.object(loop.subprocess, "run", return_value=response):
            returned = json.loads(obj.experiment({"base": "fixture", "variants": [{}], "process": job["process"]}, []))
        for metadata in returned["results"][0]["measurement_files"].values():
            copied = obj.worker_path.parent / "repo" / metadata["repo_path"]
            self.assertEqual(copied.read_bytes(), Path(metadata["path"]).read_bytes())
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), metadata["sha256"])
        self.assertEqual(returned["results"][0]["process_provenance"], captured["process_provenance"])
        unsupported = copy.deepcopy(job)
        unsupported["candidates"][0]["optimizer"]["eps"] = 1e-6
        with patch.object(kb_lab, "_gt_worker") as worker:
            with self.assertRaisesRegex(ValueError, "coupled Adam"):
                kb_lab.run_job(unsupported, workers=1, lab_dir=self.lab)
            worker.assert_not_called()
        legacy_only = dict(job)
        legacy_only.pop("capture_executable_sources")
        with self.assertRaisesRegex(ValueError, "capture_executable_sources"):
            kb_lab.run_job(legacy_only, workers=1, lab_dir=self.lab)

    def test_legacy_regression_generated_bytes_and_metrics_unchanged(self):
        sys.path.insert(0, str(kb_lab.BENCH / "src"))
        import torch
        import architecture_iq.candidates.generator as generator
        from architecture_iq.models.mlp import MlpModelFamily
        from architecture_iq.runtime.loader import load_candidate_train
        spec = {"candidate_id": "x_regression_fixture", "family": "univariate_regression",
            "model": {"type": "mlp", "input_dim": 1, "output_dim": 1, "depth": 1, "width": 4,
                      "activation": "relu", "layer_norm": [False], "residual": False},
            "optimizer": {"type": "Adam", "lr": .001}, "loss": {"loss_id": "mse"}}
        revised = self.root / "after_regression"
        generator.write_candidate(spec, revised, MlpModelFamily())
        self.assertEqual(hashlib.sha256((revised / "train.py").read_bytes()).hexdigest(),
                         "93c1bd9b6ab9d5915c259109e3d401aa69d49c4ef7e5d97bb8c830f5e5026e16")
        x = torch.tensor([[-1.], [0.], [1.]])
        y = x * x
        right = load_candidate_train(revised).train_and_eval(x, y, x, y, steps=3, batch_size=2, seed=7)
        self.assertEqual(right, {"failed": False, "final_test_mse": 0.40085139870643616,
            "eval_samples": [2, 4, 6], "step_metrics": [0.4033937454223633, 0.40258893370628357, 0.40085139870643616]})

    def test_fresh_runner_error_marks_cache_status_and_unknown_measurement(self):
        result, directory, ground_truth = self.measured_fixture({}, cached=False, error=RuntimeError("fixture failure"))
        ground_truth.assert_called_once()
        self.assertFalse(result["cached"])
        self.assertEqual(result["error"], "RuntimeError: fixture failure")
        for key in ("mean", "std", "failed_seeds", "excluded", "n_seeds", "base_seed", "seed_results"):
            self.assertIsNone(result[key])
        self.assertEqual(set(result["measurement_files"]), {"candidate_spec.json"})
        self.assertFalse((directory / "results/summary.json").exists())
        self.assertEqual(result["source_provenance"]["status"], "unverified")


if __name__ == "__main__":
    unittest.main()

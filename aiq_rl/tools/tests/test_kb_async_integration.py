import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kb_science_loop as loop
import kb_jobs


def claim(cid="K1001", text="measured comparison"):
    return {"id": cid, "text": text, "support_count": 3.0, "failure_count": 1.0,
            "credibility": 4 / 6, "created_epoch": 0, "sources": [], "merged_from": []}


class AsyncIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.obj = loop.Loop.__new__(loop.Loop)
        self.obj.run = self.run
        self.obj.lock = threading.RLock()
        self.obj.args = SimpleNamespace(model="test-model")
        self.obj.worker_path = self.run / "jobs/Jtest/job.json"
        self.obj.worker_path.parent.mkdir(parents=True)
        self.obj.worker_job = {"id": "Jtest", "origin_epoch": 3}
        self.obj.worker_path.write_text(json.dumps({**self.obj.worker_job, "payload": {}, "key": None, "status": "running"}))
        self.obj.job_sequence = 0
        self.kb = loop.KB({"epoch": 3, "claims": [claim()]}, self.run)
        self.obj.job_base = copy.deepcopy(self.kb.to_doc())
        self.kb.publish_commit = self.obj.journal_commit

    def tearDown(self):
        self.temp.cleanup()

    def research_job(self):
        self.obj.args = SimpleNamespace(model="test", research_max_cands=2, research_max_exps=1,
                                        research_turns=1, research_effort="low", corpus=None, lab=None)
        initial = self.run / "visible/initial"
        loop.write_jsonl(initial / "history.jsonl", [{"version": "initial"}])
        loop.write_json(initial / "kb.json", self.kb.to_doc())
        job = json.loads(self.obj.worker_path.read_text())
        job.update(kind="research", payload={"topic": "refresh-fixture", "question": "fixture",
                                             "kb": self.kb.to_doc(), "vis_dir": str(initial)})
        loop.write_json(self.obj.worker_path, job)
        self.obj.worker_job = job
        return initial

    def test_experiment_preserves_failed_seeds_cache_and_pinned_measurements(self):
        lab = self.run / "lab"
        source = lab / "experiments/family/safe/x_fixture/results/curves.npz"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"existing curve fixture")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        seeds = [{"seed": 51, "failed": False, "final_test_mse": 0.2},
                 {"seed": 52, "failed": True, "final_test_mse": float("inf")}]
        results = [{"mean": 0.2, "std": 0.0, "failed_seeds": 1, "excluded": True,
                    "n_seeds": 2, "base_seed": 51, "cached": True, "seed_results": seeds,
                    "measurement_files": {"results/curves.npz": {
                        "path": str(source), "sha256": digest, "bytes": source.stat().st_size}}},
                   {"mean": 0.4, "std": 0.1}]
        snapshot = source.parent / "execution_fixture"
        snapshot.mkdir()
        for name in ("model.py", "optimizer.py", "loss.py", "train.py"):
            path = snapshot / name
            path.write_text(f"EXECUTED_SOURCE = {name!r}\n")
            results[0]["measurement_files"][f"executed/{name}"] = {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        manifest = source.parent / "execution_manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "execution_dir": "results/execution_fixture"}))
        results[0]["measurement_files"]["results/execution_manifest.json"] = {"path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "bytes": manifest.stat().st_size}
        provenance = {"status": "verified", "manifest": "results/execution_manifest.json",
                      "files": [f"executed/{name}" for name in ("model.py", "optimizer.py", "loss.py", "train.py")],
                      "omitted": []}
        results[0]["source_provenance"] = provenance
        for suffix in ("json", "npz"):
            path = source.parent / "process" / ("seed_51." + suffix)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"bounded process fixture")
            results[0]["measurement_files"]["results/process/seed_51." + suffix] = {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        process_provenance = {"status": "verified", "recorder_version": "adam_ce_v1",
                              "request": {"version": "adam_ce_v1", "steps": [1], "max_eval_samples": 4}}
        results[0]["process_provenance"] = process_provenance
        base = {"family": "family", "dataset_id": "safe", "dataset": {}, "metric": "test_mse",
                "model": {"type": "mlp", "depth": 1, "width": 4}, "optimizer": {"type": "sgd"},
                "loss": {"loss_id": "mse"}, "budget": {"training_steps": 2, "batch_size": 2}}
        self.obj.args = SimpleNamespace(research_max_cands=2, lab=str(lab / "lab_db.jsonl"),
                                        lab_workers=1, pool=str(self.run / "questions.jsonl"))
        self.obj.lab_sem, self.obj.cur_epoch = threading.Semaphore(1), 3
        log = []
        response = SimpleNamespace(returncode=0, stdout=json.dumps({"dataset": "family/safe", "results": results}))
        with (patch.object(self.obj, "lab_row", return_value=base),
              patch.object(self.obj, "lab_dataset_params", return_value={}),
              patch.object(loop.subprocess, "run", return_value=response) as lab_process):
            returned = json.loads(self.obj.experiment({"base": "fixture", "variants": [{}, {}],
                                                      "process": {"steps": [1], "max_eval_samples": 4}}, log))
        submitted = json.loads(Path(lab_process.call_args.args[0][3]).read_text())
        self.assertIs(submitted["capture_executable_sources"], True)
        self.assertEqual(submitted["process"], {"steps": [1], "max_eval_samples": 4})
        first, legacy = returned["results"]
        self.assertEqual(first["seed_results"], seeds)
        self.assertEqual((first["failed_seeds"], first["excluded"], first["cached"]), (1, True, True))
        self.assertIsNone(legacy["failed_seeds"])
        self.assertIsNone(legacy["seed_results"])
        pin = first["measurement_files"]["results/curves.npz"]["repo_path"]
        self.assertEqual((self.obj.worker_path.parent / "repo" / pin).read_bytes(), source.read_bytes())
        self.assertEqual(first["source_provenance"], provenance)
        self.assertEqual(first["process_provenance"], process_provenance)
        for metadata in first["measurement_files"].values():
            captured = self.obj.worker_path.parent / "repo" / metadata["repo_path"]
            self.assertEqual(captured.read_bytes(), Path(metadata["path"]).read_bytes())
        measured = loop.read_jsonl(self.run / "lab_experiments.jsonl")
        self.assertEqual(measured[0]["seed_results"], seeds)
        self.assertEqual(measured[0]["origin_epoch"], 3)
        self.assertTrue(measured[0]["excluded"])
        self.assertEqual(measured[0]["source_provenance"], provenance)
        self.assertEqual(measured[0]["process_provenance"], process_provenance)
        evidence = next((self.obj.worker_path.parent / "repo/experiments").glob("*.json"))
        self.assertEqual(json.loads(evidence.read_text())["result"]["results"][0]["failed_seeds"], 1)
        self.assertEqual(json.loads(evidence.read_text())["result"]["results"][0]["source_provenance"], provenance)
        results[0]["measurement_files"]["results/curves.npz"]["sha256"] = "changed"
        response.stdout = json.dumps({"dataset": "family/safe", "results": results})
        with (patch.object(self.obj, "lab_row", return_value=base),
              patch.object(loop.subprocess, "run", return_value=response)):
            with self.assertRaisesRegex(ValueError, "measurement changed"):
                self.obj.experiment({"base": "fixture", "variants": [{}, {}]}, [])

    def test_research_refresh_keeps_saved_batch_before_curation_finishes(self):
        initial = self.research_job()
        loop.write_jsonl(self.run / "metrics.jsonl", [{"epoch": 2}])
        loop.write_jsonl(self.run / "epochs/e0003/records.jsonl", [{"question_id": "current-batch"}])
        seen = []

        class Session:
            log = []

            def __init__(session, workdir, model, system, tools, handler):
                session.handler = handler

            def turn(session, *args, **kwargs):
                seen.append(session.handler("refresh_observations", {}))
                return {"final": "<report>Saved training fixture only.</report>", "usage": {}}

            def close(session):
                pass

        def visible(name, records, kb):
            seen.append((name, records))
            path = self.run / "visible" / name
            loop.write_jsonl(path / "history.jsonl", records)
            loop.write_json(path / "kb.json", kb.to_doc())
            return path

        def history(epoch):
            seen.append(epoch)
            return loop.read_jsonl(self.run / "epochs" / f"e{epoch:04d}" / "records.jsonl")

        with (patch.object(loop, "CodexSession", Session),
              patch.object(self.obj, "history_records", side_effect=history),
              patch.object(self.obj, "visible", side_effect=visible),
              patch.object(self.obj, "publish_report", return_value="published")):
            self.obj.research(3, 1, "fixture", "", initial, self.kb)
        self.assertEqual(seen[0], 3)
        self.assertEqual(seen[1][1], [{"question_id": "current-batch"}])
        self.assertIn("curation may still be running", seen[2])
        self.assertEqual(json.loads(self.obj.worker_path.read_text())["observations_epoch"], 3)

    def test_refresh_same_epoch_preserves_versions_and_worker_restores_latest(self):
        initial = self.research_job()
        versions = []
        loop.write_jsonl(self.run / "metrics.jsonl", [{"epoch": 3}])

        class Session:
            log = []

            def __init__(session, workdir, model, system, tools, handler):
                session.handler = handler

            def turn(session, *args, **kwargs):
                for _ in range(2):
                    session.handler("refresh_observations", {})
                    versions.append(json.loads(self.obj.worker_path.read_text())["observations_pin"])
                return {"final": "<report>Open synthetic investigation.</report>", "usage": {}}

            def close(session):
                pass

        def visible(name, records, kb):
            path = self.run / "visible" / name
            loop.write_jsonl(path / "history.jsonl", records)
            doc = kb.to_doc()
            doc["claims"][0]["text"] = records[0]["version"]
            loop.write_json(path / "kb.json", doc)
            return path

        with (patch.object(loop, "CodexSession", Session),
              patch.object(self.obj, "history_records", side_effect=[[{"version": "first"}], [{"version": "latest"}]]),
              patch.object(self.obj, "visible", side_effect=visible),
              patch.object(self.obj, "publish_report", return_value="published")):
            self.obj.research(3, 1, "fixture", "", initial, self.kb)
        first, latest = versions
        self.assertNotEqual(first["path"], latest["path"])
        self.assertEqual([v["sequence"] for v in versions], [1, 2])
        self.assertEqual([v["epoch"] for v in versions], [3, 3])
        self.assertEqual(loop.read_jsonl(Path(first["path"]) / "history.jsonl"), [{"version": "first"}])
        self.assertEqual(self.obj.observation_files(Path(first["path"]), first["files"])[1], first["files"])
        job = json.loads(self.obj.worker_path.read_text())
        self.assertEqual(job["observations_history"], versions)
        self.assertEqual(job["payload"]["vis_dir"], str(initial))
        (self.obj.worker_path.parent / "r01.json").unlink()
        restored = loop.Loop.__new__(loop.Loop)
        restored.run, restored.lock, restored.args = self.run, threading.RLock(), self.obj.args
        seen = []

        def research(epoch, k, question, focus, vis, kb):
            seen.extend([vis, loop.read_jsonl(vis / "history.jsonl"),
                         json.loads((vis / "kb.json").read_text())["claims"][0]["text"]])
            return "Synthetic recovery complete"

        with patch.object(restored, "research", side_effect=research):
            restored.worker(self.obj.worker_path)
        self.assertEqual(seen, [Path(latest["path"]), [{"version": "latest"}], "latest"])
        self.assertEqual(json.loads(self.obj.worker_path.read_text())["status"], "completed")

    def test_observation_resume_rejects_changed_missing_and_outside_pins(self):
        initial = self.research_job()
        directory, files = self.obj.observation_files(initial)
        pin = {"path": str(directory), "epoch": 3, "files": files}
        job = json.loads(self.obj.worker_path.read_text())
        job.update(observations_path=str(directory), observations_epoch=3, observations_pin=pin)
        history = initial / "history.jsonl"
        saved = history.read_bytes()
        history.write_text("changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.obj.worker_observations(job)
        history.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.obj.worker_observations(job)
        history.write_bytes(saved)
        pin["path"] = str(self.run)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.obj.worker_observations(job)
        job["observations_pin"] = None
        job["observations_path"] = str(self.run)
        with self.assertRaisesRegex(ValueError, "outside visible"):
            self.obj.worker_observations(job)
        self.assertEqual(json.loads(self.obj.worker_path.read_text())["status"], "running")

    def test_legacy_observation_path_migrates_once_with_saved_bytes(self):
        initial = self.research_job()
        legacy = self.run / "visible/Jtest_research_e0004"
        loop.write_jsonl(legacy / "history.jsonl", [{"version": "legacy"}])
        loop.write_json(legacy / "kb.json", self.kb.to_doc())
        job = json.loads(self.obj.worker_path.read_text())
        job.update(observations_path=str(legacy), observations_epoch=4)
        loop.write_json(self.obj.worker_path, job)
        self.assertEqual(self.obj.worker_observations(job), legacy.resolve())
        migrated = json.loads(self.obj.worker_path.read_text())
        self.assertEqual(migrated["payload"]["vis_dir"], str(initial))
        self.assertEqual(migrated["observations_pin"]["epoch"], 4)
        self.assertEqual(migrated["observations_pin"]["origin_epoch"], 3)
        self.assertIn("no historical hash", migrated["observations_pin"]["migration"])
        self.obj.worker_observations(migrated)
        self.assertEqual(json.loads(self.obj.worker_path.read_text())["observations_history"],
                         [migrated["observations_pin"]])
        (legacy / "kb.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.obj.worker_observations(migrated)

    def test_empty_research_final_preserves_report_and_session_without_completion(self):
        initial = self.research_job()
        self.obj.publish_report("# Saved report\nUnresolved synthetic theory.", ["K1001"], "Fixture",
                                "refresh-fixture", self.kb)
        saved = copy.deepcopy(next(iter(self.kb.reports.values())))
        before = list((self.obj.worker_path.parent / "publications").glob("p*.json"))
        session_path = self.obj.worker_path.parent / "session/_codex_session.json"
        loop.write_json(session_path, {"thread_id": "exact-synthetic-session"})

        class Session:
            log = []
            final = ""

            def __init__(session, workdir, *args):
                session.thread_id = json.loads((workdir / "_codex_session.json").read_text())["thread_id"]

            def turn(session, *args, **kwargs):
                return {"final": session.final, "usage": {}}

            def close(session):
                pass

        for value in ["", " \n\t", "<report></report>", "prefix<report>  </report>", "(no report)"]:
            with self.subTest(final=value), patch.object(loop, "CodexSession", Session):
                Session.final = value
                with self.assertRaisesRegex(RuntimeError, "resume the same session"):
                    self.obj.worker(self.obj.worker_path)
            self.assertEqual(json.loads(session_path.read_text())["thread_id"], "exact-synthetic-session")
            self.assertEqual(Path(saved["path"]).read_text(), "# Saved report\nUnresolved synthetic theory.")
            self.assertEqual((self.obj.worker_path.parent / "repo/report.md").read_text(), Path(saved["path"]).read_text())
            self.assertEqual(list((self.obj.worker_path.parent / "publications").glob("p*.json")), before)
            self.assertNotEqual(json.loads(self.obj.worker_path.read_text())["status"], "completed")
            self.assertFalse((self.obj.worker_path.parent / "r01.json").exists())

    def test_partial_report_is_usable_and_frozen(self):
        text = "# A continuing report\nObserved evidence. Two competing explanations remain unresolved."
        result = self.obj.publish_report(text, ["K1001"], "Test research", "topic", self.kb)
        self.assertIn("published", result)
        current = loop.KB({"epoch": 7, "claims": [claim()]}, self.run)
        self.obj.apply_publications(current, 7)
        self.assertEqual(len(current.reports), 1)
        report = next(iter(current.reports.values()))
        self.assertEqual(report["origin_epoch"], 3)
        self.assertEqual(Path(report["path"]).read_text(), text)
        frozen = loop.KB(copy.deepcopy(current.to_doc()), self.run)
        self.obj.publish_report(text + "\nNew measurement.", ["K1001"], "Test research", "topic", self.kb)
        self.obj.apply_publications(current, 8)
        self.assertNotEqual(next(iter(current.reports.values()))["repo_commit"], report["repo_commit"])
        self.assertEqual(Path(next(iter(frozen.reports.values()))["path"]).read_text(), text)

    def test_late_revision_preserves_new_credits_and_is_idempotent(self):
        self.kb.merge(["K1001"], "revised scope", "new evidence", {"role": "researcher"}, None, None)
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        current.credit({"K1001": (2, 0)}, {"role": "solver"}, "q")
        self.assertEqual(len(self.obj.apply_publications(current, 6)), 1)
        self.assertEqual(current.claims["K1001"]["text"], "revised scope")
        self.assertEqual(current.claims["K1001"]["support_count"], 5)
        self.assertEqual(self.obj.apply_publications(current, 6), [])
        operations = loop.read_jsonl(self.run / "commits.jsonl")
        applied = next(c for c in operations if c["op"] == "async_apply")
        self.assertEqual((applied["origin_epoch"], applied["applied_epoch"]), (3, 6))

    def test_postrun_drain_preserves_frozen_results_and_origin_epoch(self):
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        loop.write_json(self.run / "kb_live.json", current.to_doc())
        loop.write_json(self.run / "kb/kb_0006.json", current.to_doc())
        loop.write_jsonl(self.run / "metrics.jsonl", [{"epoch": 6}])
        frozen = {path: path.read_bytes() for path in
                  (self.run / "kb/kb_0006.json", self.run / "metrics.jsonl")}
        config = {"owner": None, "owner_finished": False}
        loop.write_json(self.run / "jobs/supervisor.json", config)
        self.kb.merge(["K1001"], "late revised scope", "new evidence",
                      {"role": "researcher"}, None, None)
        with patch.object(loop, "process_alive", return_value=True):
            self.assertIsNone(loop.Loop.drain_publications(self.run))
        with patch.object(loop, "process_alive", return_value=None):
            self.assertIsNone(loop.Loop.drain_publications(self.run))
        loop.write_json(self.run / "jobs/supervisor.json", {**config, "owner_finished": True})
        with patch.object(loop, "process_alive", return_value=True):
            self.assertEqual(len(loop.Loop.drain_publications(self.run)), 1)
            self.assertEqual(loop.Loop.drain_publications(self.run), [])
        live = json.loads((self.run / "kb_live.json").read_text())
        self.assertEqual(live["epoch"], 6)
        self.assertEqual(live["claims"][0]["text"], "late revised scope")
        applied = [c for c in loop.read_jsonl(self.run / "commits.jsonl") if c["op"] == "async_apply"]
        self.assertEqual(len(applied), 1)
        self.assertEqual((applied[0]["origin_epoch"], applied[0]["applied_epoch"]), (3, 7))
        for path, contents in frozen.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_postrun_drain_recovers_pending_commit_before_exit(self):
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        current.credit({"K1001": (1, 0)}, {"role": "solver"}, "q")
        loop.write_json(self.run / "kb_live.json", current.to_doc())
        loop.write_json(self.run / "jobs/supervisor.json", {"owner_finished": False})
        with patch.object(loop, "process_alive", return_value=False):
            self.assertEqual(loop.Loop.drain_publications(self.run), [])
            self.assertEqual(loop.Loop.drain_publications(self.run), [])
        self.assertEqual(len(loop.read_jsonl(self.run / "commits.jsonl")), 1)
        self.assertEqual(json.loads((self.run / "kb_live.json").read_text())["pending_commits"], [])

    def test_worker_preserves_owner_and_main_reclaims_it(self):
        jobs = kb_jobs.Jobs(self.run, Path(loop.__file__))
        owner, worker = {"pid": 111, "started": "old"}, {"pid": 222, "started": "new"}
        loop.write_json(jobs.directory / "supervisor.json",
                        {"owner": owner, "owner_finished": True, "identity": worker})
        with (patch.object(kb_jobs, "process_alive", return_value=True),
              patch.object(kb_jobs, "process_identity", return_value=worker)):
            config = jobs.start_supervisor()
            self.assertEqual(config["owner"], owner)
            self.assertIs(config["owner_finished"], True)
            config = jobs.start_supervisor(claim_owner=True)
            self.assertEqual(config["owner"], worker)
            self.assertIs(config["owner_finished"], False)
            jobs.finish_owner()
        self.assertIs(loop.read_job(jobs.directory / "supervisor.json")["owner_finished"], True)

    def test_supervisor_drains_terminal_publications_after_owner_finished(self):
        loop.write_json(self.run / "kb_live.json",
                        loop.KB({"epoch": 6, "claims": [claim()]}, self.run).to_doc())
        self.kb.merge(["K1001"], "terminal scope", "new evidence", {"role": "researcher"}, None, None)
        loop.write_json(self.run / "jobs/supervisor.json", {"owner_finished": True})
        args = SimpleNamespace(run=self.run, program=Path(loop.__file__), timeout=2400,
                               max_workers=4, retry_limit=6, retry_delay=30)
        with (patch.object(kb_jobs.Jobs, "poll", return_value=[{"status": "completed"}]),
              patch.object(kb_jobs, "process_alive", return_value=True),
              patch.object(kb_jobs.time, "sleep") as sleep):
            kb_jobs.supervise(args)
        sleep.assert_not_called()
        self.assertEqual(json.loads((self.run / "kb_live.json").read_text())["claims"][0]["text"], "terminal scope")

    def test_worker_reads_late_report_by_topic_without_mutating_frozen_kb(self):
        frozen = loop.KB(copy.deepcopy(self.kb.to_doc()), self.run)
        first = "# Ongoing study\nCompeting explanations unresolved."
        self.obj.publish_report(first, ["K1001"], "Study", "scope-audit", self.kb)
        self.assertIn(first, self.obj.open_kb(frozen, ["scope-audit"], latest=True))
        self.obj.publish_report(first + "\nNew controlled measurement.", ["K1001"],
                                "Study", "scope-audit", self.kb)
        self.assertIn("New controlled measurement", self.obj.open_kb(frozen, ["K1001"], latest=True))
        self.assertIn("New controlled measurement", self.obj.open_kb(frozen, ["Jtest"], latest=True))
        self.assertEqual(frozen.reports, {})
        self.assertEqual(self.obj.open_kb(frozen, ["science"]), "no matching KB entries or reports")
        self.assertEqual(frozen.claims["K1001"]["text"], "measured comparison")

    def test_concurrent_add_ids_are_remapped_for_later_updates(self):
        added = self.kb.add("new observation", [], "measurement", {"role": "researcher"})
        current = loop.KB({"epoch": 6, "claims": [claim(), claim(added, "another worker")]}, self.run)
        self.obj.apply_publications(current, 6)
        self.kb.merge([added], "updated observation", "more data", {"role": "researcher"}, None, None)
        self.obj.apply_publications(current, 7)
        self.assertEqual(current.claims[added]["text"], "another worker")
        self.assertIn("updated observation", [c["text"] for c in current.claims.values()])

    def test_exact_ranking_credits_all_permutations(self):
        q = {"task": "ranking", "num_choices": 5, "answer": "A<B<C<D<E"}
        counts = set()
        for p in itertools.permutations("ABCDE"):
            _, score, inversions = loop.grade(q, "<answer>" + "<".join(p) + "</answer>")
            self.assertEqual(score, {0: 1, 1: .75, 2: .5, 3: .25}.get(inversions, 0))
            counts.add(inversions)
        self.assertEqual(counts, set(range(11)))
        self.assertEqual(loop.grade(q, "<answer>A<A<B<C<D</answer>")[1], 0)

    def test_macro_review_reserves_test_unless_explicitly_enabled(self):
        with patch.object(sys, "argv", ["loop"]):
            args = loop.parse()
        self.assertFalse(args.macro_test_readout)
        self.obj.args = args
        self.obj.split = {"test": ["heldout"]}
        loop.write_json(self.run / "kb/kb_0000.json", self.kb.to_doc())
        calls = []

        def sample(label, kb, repeats, ids=None):
            calls.append((label, ids))
            score = 0.5 if label in ("nokb", "kb_0000", "test_nokb") else 0.75
            return {"q": [score]}

        with patch.object(self.obj, "sample", side_effect=sample):
            review = self.obj.macro_review(3, self.kb)
            self.assertFalse(any(label.startswith("test_") or ids for label, ids in calls))
            self.assertNotIn("test_mean", review)
            self.assertEqual(json.loads((self.run / "best.json").read_text())["label"], "kb_0003")
            loop.write_json(self.run / "kb/kb_0003.json", self.kb.to_doc())
            calls.clear()
            args.macro_test_readout = True
            review = self.obj.macro_review(6, self.kb)
            self.assertEqual([(label, ids) for label, ids in calls if label.startswith("test_")],
                             [("test_kb_0006", ["heldout"]), ("test_nokb", ["heldout"])])
            self.assertIn("test_mean", review)

    def test_macro_rollback_requires_consecutive_bad_reviews(self):
        self.obj.args = SimpleNamespace(macro_repeats=3, macro_test_readout=False, gate_z=1)
        self.obj.split = {"test": []}
        self.kb.publish_commit = None
        loop.write_json(self.run / "kb/kb_0000.json", self.kb.to_doc())
        actions, strikes = [], []

        def sample(label, kb, repeats, ids=None):
            self.assertIsNone(ids)
            return {"fixture": [0.5]}

        # A neutral negative checkpoint interrupts a run of bad checkpoints.
        differences = [(-0.04, 0.01), (-0.005, 0.01), (-0.04, 0.01), (-0.04, 0.01)]
        for epoch, (delta, se) in enumerate(differences, 1):
            with patch.object(self.obj, "sample", side_effect=sample), \
                    patch.object(self.obj, "paired", return_value=(delta, se, 10)), \
                    patch.object(self.kb, "rollback", wraps=self.kb.rollback) as rollback:
                review = self.obj.macro_review(epoch, self.kb)
                self.assertEqual(rollback.call_count, int(epoch == 4))
            actions.append(review["action"])
            strikes.append(json.loads((self.run / "best.json").read_text())["strikes"])
        self.assertEqual(actions, ["strike 1 (keep growing)", "keep (not significantly worse)",
                                   "strike 1 (keep growing)", "rollback to kb_0000"])
        self.assertEqual(strikes, [1, 0, 1, 0])

    def test_final_test_uses_latest_without_best_and_validation_best_when_present(self):
        self.obj.args = SimpleNamespace(test_extra_kb=[], test_repeats=3)
        self.obj.split = {"test": ["heldout"]}
        for epoch in (0, 9, 100):
            loop.write_json(self.run / "kb" / f"kb_{epoch:04d}.json", {"epoch": epoch, "claims": [claim()]})
        calls = []

        def sample(label, kb, repeats, ids=None):
            calls.append((label, kb.epoch if kb else None, repeats, ids))
            return {"heldout": [0.5 if kb is None else 0.75]}

        with patch.object(self.obj, "sample", side_effect=sample):
            result = self.obj.final_test()
            self.assertEqual((result["best"], result["selection"]), ("kb_0100", "latest_checkpoint"))
            self.assertEqual(set(result["arms"]), {"nokb", "kb_0000", "kb_0100"})
            self.assertIn(("test_kb_0100", 100, 3, ["heldout"]), calls)
            loop.write_json(self.run / "best.json", {"label": "kb_0009", "epoch": 9})
            calls.clear()
            result = self.obj.final_test()
            self.assertEqual((result["best"], result["selection"]), ("kb_0009", "validation_best"))
            self.assertIn(("test_kb_0009", 9, 3, ["heldout"]), calls)
            self.assertNotIn("test_kb_0100", [call[0] for call in calls])

    def test_conflicted_merge_is_atomic_and_preserves_report(self):
        self.kb.claims["K1002"] = claim("K1002", "second condition")
        self.obj.job_base = copy.deepcopy(self.kb.to_doc())
        self.kb.merge(["K1001", "K1002"], "merged comparison", "same phenomenon",
                      {"role": "researcher"}, None, None)
        current = loop.KB({"epoch": 6, "claims": [claim(), claim("K1002", "newer conflicting evidence")]}, self.run)
        before = copy.deepcopy(current.claims)
        self.obj.apply_publications(current, 6)
        self.assertEqual(current.claims, before)
        self.assertTrue(list((self.obj.worker_path.parent).glob("conflict-*.json")))

    def test_commit_log_recovered_once_after_state_write(self):
        self.kb.merge(["K1001"], "revised scope", "new evidence", {"role": "researcher"}, None, None)
        doc = {**self.kb.to_doc(), "pending_commits": self.kb.pending}
        recovered = loop.KB(doc, self.run)
        recovered.flush()
        again = loop.KB(doc, self.run)
        again.flush()
        self.assertEqual(len(loop.read_jsonl(self.run / "commits.jsonl")), 1)

    def test_merge_transfers_new_credit_and_preserves_retired_counts(self):
        self.kb.claims["K1002"] = claim("K1002", "second condition")
        self.obj.job_base = copy.deepcopy(self.kb.to_doc())
        merged = self.kb.merge(["K1001", "K1002"], "general comparison", "controlled conditions",
                               {"role": "researcher"}, None, None)
        current = loop.KB({"epoch": 6, "claims": [claim(), claim("K1002", "second condition")]}, self.run)
        current.credit({"K1001": (2, 1)}, {"role": "solver"}, "new")
        self.obj.apply_publications(current, 6)
        self.assertEqual(current.claims[merged]["support_count"], 8)
        self.assertEqual(current.claims[merged]["failure_count"], 3)
        self.assertEqual(next(c for c in current.retired if c["id"] == "K1001")["support_count"], 5)

    def test_late_report_uses_saved_id_map_after_parent_restart(self):
        added = self.kb.add("worker observation", [], "experiment", {"role": "researcher"})
        current = loop.KB({"epoch": 6, "claims": [claim(), claim(added, "another worker")]}, self.run)
        self.obj.apply_publications(current, 6)
        current = loop.KB(json.loads((self.run / "kb_live.json").read_text()), self.run)
        target = current.publication_id_maps["Jtest"][added]
        self.obj.publish_report("# Evidence\nUnresolved theory.", [added], "Observation", "topic", self.kb)
        self.obj.apply_publications(current, 7)
        report = next(iter(current.reports.values()))
        self.assertEqual(report["claim_ids"], [target])
        self.assertIn(report["id"], current.claims[target]["report_ids"])
        self.assertNotIn("report_ids", current.claims[added])

    def test_automatic_final_report_preserves_title_and_claims(self):
        self.obj.publish_report("# First", ["K1001"], "Scoped title", "topic", self.kb)
        self.obj.publish_report("# Latest", None, None, "topic", self.kb)
        report = next(iter(self.kb.reports.values()))
        self.assertEqual(report["title"], "Scoped title")
        self.assertEqual(report["claim_ids"], ["K1001"])

    def test_invalid_multi_operation_proposal_is_not_partially_published(self):
        before = copy.deepcopy(self.kb.claims)
        result = self.obj.propose_changes(self.kb, [
            {"op": "delete", "claim_ids": ["K1001"], "reason": "refuted"},
            {"op": "revise", "claim_ids": ["K1001"], "text": "invalid", "reason": "new"}])
        self.assertIn("rejected", result)
        self.assertEqual(self.kb.claims, before)
        self.assertFalse(list((self.obj.worker_path.parent / "publications").glob("p*.json")))

    def test_worker_commit_counter_survives_checkpoint(self):
        self.kb.add("first", [], "experiment", {"role": "researcher"})
        self.kb.merge(["K1001"], "new scope", "experiment", {"role": "researcher"}, None, None)
        saved = json.loads((self.obj.worker_path.parent / "kb_checkpoint.json").read_text())
        recovered = loop.KB(saved["kb"], self.obj.worker_path.parent)
        self.assertEqual(recovered.counter, 2)
        recovered.add("third", [], "experiment", {"role": "researcher"})
        self.assertEqual(recovered.pending[-1]["commit"], "c000003")

    def test_parallel_report_publications_are_serialized(self):
        errors = []
        def publish(i):
            try:
                self.obj.publish_report(f"# Evidence {i}", ["K1001"], "Scope", "topic", self.kb)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=publish, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertFalse(errors)
        publications = list((self.obj.worker_path.parent / "publications").glob("p*.json"))
        self.assertEqual(len(publications), 4)
        self.assertEqual(len({json.loads(p.read_text())["operation"]["commit"] for p in publications}), 4)

    def test_frozen_solve_credit_follows_merge_after_parent_recovery(self):
        current = loop.KB({"epoch": 6, "claims": [claim(), claim("K1002", "second condition")]}, self.run)
        target = current.merge(["K1001", "K1002"], "general comparison", "measured", {}, None, None)
        current.credit({"K1001": (1, 0), "K1002": (1, 0)}, {}, "frozen_question")
        self.assertEqual(current.claims[target]["support_count"], 8)
        current.credit({"K1001": (1, 0)}, {}, "frozen_question")
        self.assertEqual(current.claims[target]["support_count"], 8)

    def test_late_deletion_cannot_discard_new_science_field(self):
        self.kb.delete("K1001", "older rejection", {"role": "researcher"})
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        current.claims["K1001"]["mechanism"] = "New controlled intervention"
        self.obj.apply_publications(current, 6)
        self.assertIn("K1001", current.claims)
        self.assertEqual(current.claims["K1001"]["mechanism"], "New controlled intervention")

    def test_deferred_new_claim_keeps_identity_for_late_reports(self):
        self.obj.propose_changes(self.kb, [
            {"op": "revise", "claim_ids": ["K1001"], "text": "older scope", "reason": "test"},
            {"op": "add", "text": "local discovery", "reason": "test"}])
        current = loop.KB({"epoch": 6, "claims": [claim(text="newer scope"), claim("K1002", "unrelated")]}, self.run)
        self.obj.apply_publications(current, 6)
        current = loop.KB(json.loads((self.run / "kb_live.json").read_text()), self.run)
        mapped = current.publication_id_maps["Jtest"]["K1002"]
        self.assertNotEqual(mapped, "K1002")
        self.obj.publish_report("# Followup", ["K1002"], "Followup", "topic", self.kb)
        self.obj.apply_publications(current, 7)
        report = next(iter(current.reports.values()))
        self.assertEqual(report["claim_ids"], [])
        self.assertEqual(report["deferred_claim_ids"], [mapped])
        self.assertNotIn("report_ids", current.claims["K1002"])
        self.assertIn("Followup", self.obj.open_kb(current, ["science"]))

    def test_chained_merge_preserves_colliding_intermediate_lineage(self):
        self.kb.claims.update({"K1002": claim("K1002"), "K1003": claim("K1003")})
        self.obj.job_base = copy.deepcopy(self.kb.to_doc())
        self.obj.propose_changes(self.kb, [
            {"op": "merge", "claim_ids": ["K1001", "K1002"], "text": "first", "reason": "test"},
            {"op": "merge", "claim_ids": ["K1004", "K1003"], "text": "final", "reason": "test"}])
        current = loop.KB({"epoch": 6, "claims": [claim(), claim("K1002"), claim("K1003"),
                                                    claim("K1004", "unrelated")]}, self.run)
        current.credit({"K1001": (2, 0)}, {}, "new")
        self.obj.apply_publications(current, 6)
        mapping = current.publication_id_maps["Jtest"]
        intermediate, final = mapping["K1004"], mapping["K1005"]
        retired = {c["id"]: c for c in current.retired}
        self.assertEqual(retired["K1001"]["merged_into"], intermediate)
        self.assertEqual(retired[intermediate]["merged_into"], final)
        self.assertEqual(retired[intermediate]["support_count"], 8)
        self.assertEqual(current.claims[final]["merged_from"], [intermediate, "K1003"])
        self.assertEqual(current.claims[final]["support_count"], 11)
        self.assertEqual(current.claims["K1004"]["text"], "unrelated")
        current = loop.KB(json.loads((self.run / "kb_live.json").read_text()), self.run)
        current.credit({"K1001": (1, 0)}, {}, "frozen")
        self.assertEqual(current.claims[final]["support_count"], 12)

    def test_older_topic_report_cannot_overwrite_newer_saved_version(self):
        self.obj.publish_report("# Newer", ["K1001"], "Topic", "topic", self.kb)
        path = next((self.obj.worker_path.parent / "publications").glob("p*.json"))
        newer = json.loads(path.read_text())
        report_id = next(iter(newer["after"]["reports"]))
        newer["published_at"] = "2026-10-04T11:00:00+00:00"
        newer["after"]["reports"][report_id]["published_at"] = newer["published_at"]
        path.write_text(json.dumps(newer))
        older = copy.deepcopy(newer)
        older["id"], older["job_id"] = "Jzold:000001", "Jzold"
        older["published_at"] = "2026-10-04T10:00:00+00:00"
        older["after"]["reports"][report_id]["published_at"] = older["published_at"]
        older["after"]["reports"][report_id]["title"] = "Older"
        older_path = self.run / "jobs/Jzold/publications/p000001.json"
        older_path.parent.mkdir(parents=True)
        older_path.write_text(json.dumps(older))
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        current.reports = copy.deepcopy(newer["after"]["reports"])
        self.obj.apply_publications(current, 6)
        self.assertEqual(current.reports[report_id]["title"], "Topic")


if __name__ == "__main__":
    unittest.main()

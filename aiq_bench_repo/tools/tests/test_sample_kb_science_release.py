"""Synthetic fixtures validate release mechanics; they are never published."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "sample_kb_science_release.py"
SPEC = importlib.util.spec_from_file_location("sample_kb_science_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def question(source, i, member=0):
    ranking = source.startswith("ranking")
    q = {"question_id": f"{source}_{i}_{member}", "source": source, "group": f"{source}_group_{i}",
         "family": "synthetic_test_only", "task": "ranking" if ranking else "select",
         "num_choices": 5 if ranking else 3, "answer": "A<B<C<D<E" if ranking else "A",
         "messages": [{"role": "user", "content": "Synthetic fixture for release mechanics."}],
         "meta": {"gate": "synthetic_test_only"}}
    if source in ("arch170", "ranking_v3") and i < 6:
        q["meta"]["candidate_set"] = f"datasets/synthetic/shared_{i}/candidates/set_{source}"
    return q


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = self.root / "synthetic.jsonl"
        self.questions = [question(s, i, member) for s in release.SOURCES for i in range(40)
                          for member in range(2 if s == "dataflip500" else 1)]
        self.write_pool()
        self.sources = {s: self.pool for s in release.SOURCES}

    def tearDown(self):
        self.temp.cleanup()

    def write_pool(self):
        self.pool.write_text("".join(json.dumps(q) + "\n" for q in self.questions))

    def build(self, directory="release", counts=None, exclusions=()):
        target = self.root / directory
        manifest = release.build(self.sources, counts or {s: 20 for s in release.SOURCES},
                                 exclusions, target, seed=123, name="synthetic_test_only")
        return target, manifest

    def test_replay_and_counts_with_complete_dataflip_pairs(self):
        first, manifest = self.build("a")
        second, _ = self.build("b")
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertEqual(set(manifest["files"]), set(release.PAYLOAD_FILES))
        self.assertEqual(manifest["ranking_credit"], {"0": 1.0, "1": 0.75, "2": 0.5, "3": 0.25})
        self.assertEqual(manifest["other_ranking_credit"], 0)
        split_doc = json.loads((first / "split.json").read_text())
        assigned = {qid: split for split in release.SPLITS for qid in split_doc[split]}
        pool, _ = release.rows(first / "questions.jsonl")
        pairs = {}
        for q in pool:
            if q["source"] == "dataflip500":
                pairs.setdefault(q["group"], []).append(q)
        self.assertTrue(pairs)
        for members in pairs.values():
            self.assertEqual(len(members), 2)
            self.assertEqual(len({assigned[q["question_id"]] for q in members}), 1)
        for source in release.SOURCES:
            count = manifest["sources"][source]["selected_questions"]
            self.assertGreaterEqual(count, 20)
            self.assertLessEqual(count, 26)
        self.assertEqual(release.verify(first), manifest)

    def test_cross_source_dataset_identity_and_seen_record_stay_train(self):
        seen = self.root / "seen.jsonl"
        seen.write_text(json.dumps({"question_id": "arch170_0_0"}) + "\n")
        counts = {s: 10000 for s in release.SOURCES}
        path, manifest = self.build(counts=counts, exclusions=[seen])
        split_doc = json.loads((path / "split.json").read_text())
        groups = split_doc["group_by_question"]
        for i in range(6):
            self.assertEqual(groups[f"arch170_{i}_0"], groups[f"ranking_v3_{i}_0"])
        group = split_doc["groups"][groups["arch170_0_0"]]
        self.assertTrue(group["seen_before"])
        self.assertEqual(group["split"], "train")
        self.assertIn("ranking_v3_0_0", split_doc["train"])
        self.assertTrue(split_doc["val"])
        self.assertTrue(split_doc["test"])
        self.assertLess(manifest["sources"]["arch170"]["difference_from_requested"], 0)

    def test_malformed_gold_and_conflicting_duplicate_fail_before_output(self):
        self.questions[40]["answer"] = "A<A<C<D<E"
        self.write_pool()
        with self.assertRaisesRegex(ValueError, "invalid gold"):
            self.build()
        self.assertFalse((self.root / "release").exists())
        self.questions[40]["answer"] = "A<B<C<D<E"
        conflict = dict(self.questions[0], answer="B")
        self.questions.append(conflict)
        self.write_pool()
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.build()

    def test_identical_duplicates_are_explicitly_deduplicated(self):
        self.questions.append(dict(self.questions[0]))
        self.write_pool()
        path, manifest = self.build()
        self.assertEqual(manifest["sources"]["arch170"]["identical_duplicates_removed"], 1)
        self.assertEqual(manifest["sources"]["arch170"]["input_questions"], 40)
        self.assertEqual(release.verify(path), manifest)

    def test_missing_input_and_existing_release_refuse(self):
        self.sources["ranking_v3"] = self.root / "absent.jsonl"
        with self.assertRaisesRegex(ValueError, "cannot read JSONL"):
            self.build()
        self.assertFalse((self.root / "release").exists())
        self.sources["ranking_v3"] = self.pool
        path, _ = self.build()
        before = (path / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.build()
        self.assertEqual(before, (path / "manifest.json").read_bytes())

    def test_cli_and_hash_tampering(self):
        target = self.root / "cli"
        command = [sys.executable, str(SCRIPT), "--output", str(target), "--name", "synthetic_test_only"]
        for source in release.SOURCES:
            command.extend(["--source", f"{source}={self.pool}", "--sample", f"{source}=20"])
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(json.loads(result.stdout)["questions"], 0)
        check = subprocess.run([sys.executable, str(SCRIPT), "--verify", str(target)], text=True, capture_output=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        with (target / "README.md").open("a") as handle:
            handle.write("tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            release.verify(target)

    def test_run_directory_and_plain_id_exclusions(self):
        run = self.root / "old_run"
        epoch = run / "epochs/e0001"
        epoch.mkdir(parents=True)
        (epoch / "records.jsonl").write_text(json.dumps({"question_id": "dataflip500_0_0"}) + "\n")
        ids = self.root / "ids.txt"
        ids.write_text("arch170_0_0\nranking_v2_0_0\n")
        path, manifest = self.build(counts={s: 10000 for s in release.SOURCES}, exclusions=[run, ids])
        split_doc = json.loads((path / "split.json").read_text())
        self.assertTrue({"dataflip500_0_0", "dataflip500_0_1", "arch170_0_0",
                         "ranking_v3_0_0", "ranking_v2_0_0"}.issubset(set(split_doc["train"])))
        self.assertEqual(manifest["seen_records"], 3)

    def test_semantic_verifier_rejects_inconsistent_split_count(self):
        path, manifest = self.build()
        manifest["sources"]["arch170"]["split_counts"]["train"] += 1
        (path / "manifest.json").write_bytes(release.json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "split counts differ"):
            release.verify(path)

    def test_json_id_list_and_non_finite_fraction(self):
        seen = self.root / "ids.json"
        seen.write_text(json.dumps(["ranking_v2_0_0"]))
        path, _ = self.build(counts={s: 10000 for s in release.SOURCES}, exclusions=[seen])
        self.assertIn("ranking_v2_0_0", json.loads((path / "split.json").read_text())["train"])
        with self.assertRaisesRegex(ValueError, "fractions"):
            release.build(self.sources, {s: 20 for s in release.SOURCES}, [], self.root / "bad", val_fraction=float("nan"))

    def test_exhausted_holdout_stays_train_without_losing_source_total(self):
        seen = self.root / "seen.jsonl"
        seen.write_text("".join(json.dumps({"question_id": q["question_id"]}) + "\n"
                                for q in self.questions if q["source"] == "arch170"))
        path, manifest = self.build(exclusions=[seen])
        split = json.loads((path / "split.json").read_text())
        selection = json.loads((path / "selection.json").read_text())
        self.assertEqual(manifest["sources"]["arch170"]["selected_questions"], 20)
        self.assertEqual(manifest["sources"]["arch170"]["split_counts"],
                         {"train": 20, "val": 0, "test": 0})
        self.assertEqual(selection["holdout_shortfalls"]["arch170"], {"val": 3, "test": 3})
        self.assertTrue(all(g["split"] == "train" for g in split["groups"].values()
                            if g["source_counts"].get("arch170")))

    def test_dataset_seed_identity_connects_renamed_groups(self):
        self.questions[0]["meta"]["dataset_seed_identity"] = "fixture-family:point_seed=123"
        self.questions[80]["meta"]["dataset_seed_identity"] = "fixture-family:point_seed=123"
        self.questions[80]["meta"].pop("candidate_set", None)
        self.write_pool()
        seen = self.root / "seen.jsonl"
        seen.write_text(json.dumps({"question_id": self.questions[0]["question_id"]}) + "\n")
        path, _ = self.build(counts={s: 10000 for s in release.SOURCES}, exclusions=[seen])
        split = json.loads((path / "split.json").read_text())
        first, second = self.questions[0]["question_id"], self.questions[80]["question_id"]
        self.assertEqual(split["group_by_question"][first], split["group_by_question"][second])
        self.assertTrue({first, second}.issubset(split["train"]))

    def test_source_provenance_is_frozen_and_hash_verified(self):
        provenance = self.root / "inventory.json"
        inventory = {"seed_boundary": "unknown legacy seed exposure", "sources": []}
        provenance.write_text(json.dumps(inventory))
        output = self.root / "release"
        release.build(self.sources, {s: 20 for s in release.SOURCES}, [], output, provenance=provenance)
        self.assertEqual(json.loads((output / "source_provenance.json").read_text()), inventory)
        release.verify(output)
        (output / "source_provenance.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            release.verify(output)


if __name__ == "__main__":
    unittest.main()

# Science of AI KB

A research prototype that grows an explicit knowledge base while a frozen model solves neural network training questions. Solvers receive short claims and can open linked reports; researchers and curators run as persistent jobs, with changes attributed to their originating epoch.

This release includes the runtime, concrete KB, 15 epochs of saved training records, and controlled studies. Version 0.2 adds a directed research program with 12 studies and seven concise claims. It reproduces these artifacts without establishing a general training theory or a significant held-out score gain.

## Directed science

[Study index and findings](programs/directed_science/README.md) links the residual, effective-time/capacity and activation/generalization research chains. Reports include failed predictions, prospective OOD conditions, archived execution sources and evidence limits. No new solver evaluation was run.

After setup, download the versioned raw measurements and recompute all saved analyses without training or model calls:

~~~sh
python tools/verify_science_program.py --download
~~~

Three release assets carry the original numerical arrays and per-cell records. The repository contains their SHA256 manifest, predictions and analyses; the verifier refuses changed arrays or mismatched statistics. See the [program handoff](programs/directed_science/HANDOFF.md).

## Inspect the results

- [Interactive experiment and KB](https://guoshaoyang-pku.github.io/blogs/kb_site/runs/soa_async_v1.html#kb=latest_saved): snapshots, per-question records, colored add/merge/delete history and reports.
- [Latest concrete KB](examples/soa_async_v1/kb_live.json): 143 claims and 77 report references. The evaluated epoch 15 snapshot has 139 claims.
- [Label mean intervention](examples/soa_async_v1/jobs/Jfc41e23dc91c/repo/report.md) and [initial output / frozen hidden controls](examples/soa_async_v1/jobs/Jfab91d105641/repo/report.md): 600 saved controlled training runs.
- [Handoff](docs/HANDOFF.md): contracts, evidence boundaries and continuation constraints.

On 105 development validation ranking questions with three repeats, epoch 15 scored **70.08%**, versus **65.87%** without KB: **+4.21 ± 2.37 percentage points**, paired question-level 1 SE. Scores include partial ranking credit. Validation was used during development. Final benchmark test questions are excluded; controlled studies produced no new solver measurement.

## Run without a model key

Python 3.10+ on Linux or macOS is required. Setup installs the included ArchitectureIQ runtime, MCP bridge and test dependencies.

~~~sh
git clone https://github.com/guoshaoyang-pku/science-of-ai-kb.git
cd science-of-ai-kb
./setup.sh
. .venv/bin/activate
python tools/verify_research.py
python tools/render_example.py --output build/soa_async_v1.html
python -m http.server 8000 --directory build
~~~

Open http://127.0.0.1:8000/soa_async_v1.html. Verification recomputes statistics in a temporary copy without training or API calls. The viewer includes short claims and report text; research data stays in its directories instead of being embedded into huge HTML files.

## Run new epochs

Supply your own question release and model endpoint. Copy the dummy credential file to the ignored location, replace endpoint/key privately, and export its path. The scripts do not automatically load .env.

~~~sh
cp aiq_bench_repo/data/evals/eval_keys.example.json aiq_bench_repo/data/evals/eval_keys.json
export KB_EVAL_KEYS="$PWD/aiq_bench_repo/data/evals/eval_keys.json"
python aiq_rl/tools/kb_science_loop.py --help
python aiq_rl/tools/kb_science_loop.py \
  --run-name my_kb_run --pool /path/to/questions.jsonl \
  --provider vapi --model YOUR_MODEL --summ-model YOUR_MODEL \
  --solver-backend api --summ-backend api \
  --seed-kb examples/empty_kb.json --render brief --science \
  --batch-size 30 --epochs 5 --macro-every 5 --macro-repeats 3 \
  --train-control --test-repeats 0
~~~

For persistent asynchronous research, install a compatible Codex CLI and select --solver-backend codex --summ-backend codex --async-agents. Codex requires a Responses endpoint; the API adapter uses Chat Completions. Optional Claude Code needs its CLI and compatible endpoint. Access and billing depend on your environment; offline tests use fake CLIs.

Measured lab data is supplied separately. The lab builder excludes held-out groups and synthesis seeds and writes a hashed allowlist, validated before training. Rebuild it for your release. Examples are an inspection archive, not a continuation of the original run or replacement for a question release.

## Controlled-study finding

At one MLP/SGD/Adam recipe and three fixed regression functions, centering labels favored Adam; offsets of ±3 favored SGD. Shifting initial output bias by the same amount removed the reversal. Freezing hidden layers restored the affine output response and convex endpoint loss in label offset predicted by fixed-feature SGD.

The useful decision variable is initial target/output mean mismatch. Plain MSE with additive output bias has a broader translation symmetry when bias is not decayed; the measured recipe has weak coupled decay and is approximately invariant. Hidden/head mediation and new-recipe prediction remain unresolved.

Machine paths in metadata were normalized. [Release scope](examples/soa_async_v1/release_scope.json) and each study's release_metadata_changes.json distinguish original identities from public hashes. Numerical arrays and archived executed training sources are byte-identical. Public metadata hashes do not claim byte identity with private originals.

## Develop

~~~sh
python -m pytest aiq_rl/tools/tests aiq_rl/tools/test_kb_jobs.py \
  aiq_bench_repo/tools/tests/test_sample_kb_science_release.py aiq_bench_repo/tests -q
~~~

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Codex and Claude Code share this guide. Python isolation differs by OS and is not a guarantee for hostile code on arbitrary hosts.

MIT licensed, preserving ArchitectureIQ Authors (MetaCircle) attribution. The included runtime is a subset of [ArchitectureIQ](https://github.com/renrua52/ArchitectureIQ).

# Repository guide

## Commands

~~~sh
./setup.sh
. .venv/bin/activate
python tools/verify_research.py
python tools/render_example.py --output build/soa_async_v1.html
python -m pytest aiq_rl/tools/tests aiq_rl/tools/test_kb_jobs.py aiq_bench_repo/tools/tests/test_sample_kb_science_release.py aiq_bench_repo/tests -q
~~~

## Entry points

| Path | Responsibility |
|---|---|
| aiq_rl/tools/kb_science_loop.py | Solver/discovery/curation, KB operations, short claims, expansion, checkpoint review |
| aiq_rl/tools/kb_jobs.py | Persistent jobs, identities, locks and recovery |
| aiq_rl/tools/codex_agent.py, cc_agent.py, kb_mcp_bridge.py | Continuing CLI sessions and controlled tool bridge |
| aiq_rl/tools/kb_lab.py, kb_sandbox.py | Allowlist, executable capture, training and Python isolation |
| aiq_rl/docs/kb_viewer/ | Six views, operation colors and lineage |
| aiq_rl/tools/kb_publish.py | Publication recovery and HTTP/hash verification |
| aiq_bench_repo/src/architecture_iq/ | Included benchmark runtime and process recorder |
| examples/soa_async_v1/ | Saved records, snapshots, reports and portable studies |

## Contracts

- Freeze solver inputs and exclude revealed answers/history. Never solve from a gold-bearing conversation.
- Preserve successful solutions, caches, report versions and explicit failures.
- Async changes belong to origin_epoch; record application separately. Saved science may be used without a ready gate.
- Research may branch and span epochs. Task count alone does not establish inefficiency.
- Revise/merge/withdraw using evidence; retain raw measurements and conflicts. Credit is usage feedback, not truth probability.
- Preserve dataset/group/seed exclusion and allowlist hashes. Do not expose final test to agents.
- Separate development validation gains, study predictions and solver efficacy.
- Preserve homepage layout. Unwrap async operations before coloring; report commits are not claim additions.
- Never commit credentials, CLI homes, raw sessions or process metadata. KB_EVAL_KEYS points to private JSON.
- This archive cannot resume the original run without release/lab manifests. Async stream_offset is ignored and does not establish unseen questions.

Read [docs/HANDOFF.md](docs/HANDOFF.md) before a new goal. Keep changes scoped to requested behavior.

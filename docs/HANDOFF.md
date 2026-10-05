# Prototype handoff · 2026-10-06

This is the completed inspection/reproduction snapshot of the KB prototype. A new experimental objective should start from its contracts and evidence, without assuming that the original question stream or private model environment is bundled.

## Saved state

| Item | State |
|---|---|
| Formal run | soa_async_v1, 15 completed epochs, batch 30 |
| Training solutions | 450 with KB, 450 noKB controls |
| Async agents | 72 completed research jobs, 15 completed curators, 3 completed local analyses |
| Knowledge | live 143 claims / 77 reports; evaluated e15 snapshot 139 claims |
| Macro15 | KB .7008, noKB .6587, +.0421 ± .0237 paired question-level SE |
| Validation scope | 105 ranking_v2/v3 development questions, 3 repeats |
| Controlled research | e16 240 runs, e17 360 runs; no new solver evaluation |

The run's local-analysis tasks and historical research reports are retained as saved outputs. Only the two causal studies include their complete portable measurement repos. Imported training histories are not fully redistributed. No final benchmark test question, answer or evaluation trace is included.

## Execution contracts

Default input is the complete short KB; kb_open expands several categories, claims or reports at once. Reports can span claims, retain competing explanations and remain unfinished while the argument stays reviewable. Saved science can be used immediately; there is no ready gate.

Solvers use frozen inputs. Discovery reveals gold afterwards and continues the conversation. Curators/researchers persist across epochs, publish versioned reports and submit atomic KB changes. origin_epoch owns their changes; applied_epoch records when a change reaches live KB. Research may branch naturally without an additional strict topic cap. Raw evidence survives withdrawal of a claim or mechanism.

Batch 30 remains the current unit. Macro review examines growth over several epochs rather than selecting every small change with the same short validation set. Citation credit is task feedback, not a calibrated probability that a claim is true.

## Studies and useful knowledge

e16 Jfc41e23dc91c / R613055ebce, original research commit 44b021dec5f5349960da03162810a50493d192b4: shift labels at three fixed functions, pairing initialization, inputs and minibatch streams. Mean 0 favors Adam; means ±3 favor SGD. Nine preregistered directions matched, each with ten seeds. Endpoint centered shape fitting accounts for almost all the ranking reversal; offset residual explains at most about 5.3%.

e17 Jfab91d105641 / R7ed4fd7139, original research commit 11ae7f00d26614085894dbdfd614043c700f358a: shifting initial output bias together with labels removes the reversal, with maximum endpoint MSE change 1.00823e-5. Freezing hidden layers restores affine outputs and nonnegative SGD chord gaps for all 30 seeds. This is convex endpoint loss in the label offset, not global network convexity. All 109 finite high-loss endpoints remain present.

Recipe: 8D plain MLP, depth3/width192, GELU, LN010, no residual; plain MSE; SGD .001/momentum0 versus coupled Adam3e-5/(.9,.999); wd1e-4; 256 steps/batch64; seeds0–9. All intervention directions belong to the same three functions, not nine independent new functions.

The broader principle is output translation symmetry: simultaneous label and initial additive-bias shifts preserve MSE residual/gradient paths when bias is not decayed. Weak coupled decay makes the measured example approximately symmetric. Initial mean residual is a better decision variable than absolute target mean. LN, feature/Jacobian drift and head/hidden feedback have not been separated. Predictive transfer to unseen recipes/functions remains open.

## Reproduce and inspect

Run the README setup, then tools/verify_research.py and tools/render_example.py. They read only saved results. Study repositories contain archived executable source, paired tensors, preregistration, raw outputs and analyses. e16/reproduce_cell.py can explicitly retrain one cell if requested; original run_study.py entries are provenance scripts requiring the original scheduler/lab environment, not standalone fresh-clone commands.

Public metadata paths were normalized. Each study's release_metadata_changes.json maps changed metadata to its original/public fingerprint. The e16 evidence manifest preserves original identity keys while verifying public bytes; source_sha256 handles changed metadata references. Numerical arrays and executed training code are unchanged. The public Git release commit is distinct from the original research commits.

The epoch viewer restores green add, red delete and yellow merge/revise. Async applications unwrap their inner operations, proposal status is shown explicitly, and merge lineage uses canonical remapped IDs. Git commits for reports remain separate from KB structure operations.

## Continue with a new objective

The original training stream has only 60 unseen ranking_v2 questions left; other sources are exhausted. Its inputs are not bundled. Creating another run or changing a seed can repeat exposed data; async stream_offset currently does not advance the stream. Build a new immutable release with group/seed/variant boundaries and a historical-exposure inventory before claiming more unseen solver training.

Do not rerun successful solutions, change historical frozen snapshots, load leaked old corpora, reveal final test to agents, or add blocks to the existing homepage. The current score does not establish that strong models already know all architecture intuition. No matched Opus/sol curator comparison establishes relative capability.

Useful follow-up studies separate head from hidden trainability, measure feature/Jacobian drift and preregister predictions for new recipes. A separate new solver evaluation is required to turn scientific knowledge into a score claim. Local process IDs, credentials, publisher checkout and monitor state belong in a private operational handoff, not this release. Automatic current-turn steer/wakeup was not verified.

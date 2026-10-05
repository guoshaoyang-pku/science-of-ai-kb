# ArchitectureIQ runtime

This directory contains the ArchitectureIQ source runtime required by the KB
science prototype, including candidate generation, controlled training, dataset
plugins, profiles and prompt templates. It is an MIT-licensed subset of
[ArchitectureIQ](https://github.com/renrua52/ArchitectureIQ).

From the prototype root, install with:

```sh
python -m pip install -e ./aiq_bench_repo
```

Question releases and measured dataset artifacts are supplied separately.
The full benchmark UI and unrelated deployment services are not included.

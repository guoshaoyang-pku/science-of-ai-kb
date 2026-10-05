#!/usr/bin/env python3
"""Verify public evidence and recompute saved study statistics without training."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / 'examples/soa_async_v1/jobs'
IDENTIFIERS = ['Jfc41e23dc91c', 'Jfab91d105641']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    first = STUDIES / IDENTIFIERS[0] / 'repo'
    manifest = json.loads((first / 'evidence_manifest.json').read_text())
    for metadata in manifest['files'].values():
        path = first / metadata['path']
        assert digest(path) == metadata['sha256'], path.name
        assert path.stat().st_size == metadata['bytes'], path.name
    second = STUDIES / IDENTIFIERS[1] / 'repo'
    for name, metadata in json.loads((second / 'source_manifest.json').read_text()).items():
        assert digest(second / name) == metadata['sha256'], name
    for metadata in json.loads((second / 'base_manifest.json').read_text())['files'].values():
        assert digest(second / metadata['path']) == metadata['sha256']
    for repo in [first, second]:
        for metadata in json.loads((repo / 'release_metadata_changes.json').read_text()):
            assert digest(repo / metadata['file']) == metadata['public_sha256'], metadata['file']
    result = {'evidence_files_verified':len(manifest['files']), 'studies':[]}
    with tempfile.TemporaryDirectory(prefix='kb-research-verify-') as temporary:
        for identifier in IDENTIFIERS:
            source = STUDIES / identifier / 'repo'
            target = Path(temporary) / identifier / 'repo'
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            expected = json.loads((source / 'analysis.json').read_text())
            process = subprocess.run([sys.executable, str(target / 'analyze_results.py')],
                                     cwd=target, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stderr[-2000:])
            actual = json.loads((target / 'analysis.json').read_text())
            assert actual == expected, identifier + ': recomputed analysis differs'
            result['studies'].append({'job':identifier, 'seed_runs':actual['seed_runs'],
                                     'analysis_matches_saved':True})
    result.update(training=False, api_calls=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

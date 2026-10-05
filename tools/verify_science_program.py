#!/usr/bin/env python3
"""Fetch pinned measurements or recompute all directed-science analyses offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / 'programs/directed_science'


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest(base, files):
    for name, pin in files.items():
        path = base / name
        if not path.is_file():
            raise RuntimeError('Missing measurement: ' + name + '; run with --download first')
        if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
            raise RuntimeError('Measurement fingerprint mismatch: ' + name)


def download(manifest):
    for asset in manifest['assets']:
        files = {name: manifest['files'][name] for name in asset['files']}
        if all((PROGRAM / name).is_file() and sha(PROGRAM / name) == pin['sha256'] for name, pin in files.items()):
            print(asset['name'] + ': already verified', flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix='science-download-') as temporary:
            archive = Path(temporary) / asset['name']
            parts = asset.get('parts')
            if parts:
                with archive.open('wb') as assembled:
                    for part in parts:
                        fragment = Path(temporary) / part['name']
                        print('Downloading ' + part['name'], flush=True)
                        urllib.request.urlretrieve(part['url'], fragment)
                        if fragment.stat().st_size != part['bytes'] or sha(fragment) != part['sha256']:
                            raise RuntimeError('Release part fingerprint mismatch: ' + part['name'])
                        with fragment.open('rb') as source:
                            shutil.copyfileobj(source, assembled)
                        fragment.unlink()
            else:
                print('Downloading ' + asset['name'], flush=True)
                urllib.request.urlretrieve(asset['url'], archive)
            if archive.stat().st_size != asset['bytes'] or sha(archive) != asset['sha256']:
                raise RuntimeError('Release asset fingerprint mismatch: ' + asset['name'])
            unpacked = Path(temporary) / 'unpacked'
            unpacked.mkdir()
            with tarfile.open(archive, 'r:gz') as bundle:
                members = bundle.getmembers()
                if {m.name for m in members} != set(files) or any(not m.isfile() for m in members):
                    raise RuntimeError('Asset file list differs from the pinned manifest')
                for member in members:
                    relative = Path(member.name)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise RuntimeError('Unsafe release member')
                    target = unpacked / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
            verify_manifest(unpacked, files)
            for name, pin in files.items():
                destination = PROGRAM / name
                if destination.exists():
                    if sha(destination) != pin['sha256']:
                        raise RuntimeError('Existing measurement differs; refusing overwrite: ' + name)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(unpacked / name, destination)
            print(asset['name'] + ': verified', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='fetch the versioned evidence assets; no training or model calls')
    parser.add_argument('--hash-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((PROGRAM / 'measurements.json').read_text())
    if args.download:
        download(manifest)
    verify_manifest(PROGRAM, manifest['files'])
    source_manifest = json.loads((PROGRAM / 'sources/manifest.json').read_text())
    verify_manifest(PROGRAM, {row['file']: row for row in source_manifest['files']})
    changes = json.loads((PROGRAM / 'release_metadata_changes.json').read_text())
    for row in changes:
        if sha(PROGRAM / row['file']) != row['public_sha256']:
            raise RuntimeError('Public metadata fingerprint mismatch: ' + row['file'])
    result = {'measurement_files': len(manifest['files']), 'source_files': len(source_manifest['files']), 'training': False, 'model_calls': 0, 'analyses': []}
    if not args.hash_only:
        with tempfile.TemporaryDirectory(prefix='science-recompute-') as temporary:
            target = Path(temporary) / 'directed_science'
            shutil.copytree(PROGRAM, target, copy_function=shutil.copy2, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.lock'))
            for study in manifest['studies']:
                folder = target / 'studies' / study['name']
                for analysis in study['analyses']:
                    expected = json.loads((folder / analysis['output']).read_text())
                    command = [sys.executable, str(folder / 'analyze.py'), *analysis.get('args', [])]
                    process = subprocess.run(command, cwd=target, capture_output=True, text=True)
                    if process.returncode:
                        raise RuntimeError(study['name'] + ': analysis failed: ' + process.stderr[-2500:])
                    actual = json.loads((folder / analysis['output']).read_text())
                    if actual != expected:
                        raise RuntimeError(study['name'] + ': analysis differs from saved evidence')
                    result['analyses'].append(study['name'] + '/' + analysis['output'])
            for direction in ['B_effective_time_capacity', 'C_activation_generalization']:
                process = subprocess.run([sys.executable, str(target / 'directions' / direction / 'verify.py')], cwd=target, capture_output=True, text=True)
                if process.returncode:
                    raise RuntimeError(direction + ': contracts/spectra failed: ' + process.stderr[-2500:])
            result['direction_contracts_and_spectra'] = 'passed'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

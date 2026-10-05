#!/usr/bin/env python3
"""Build the saved example viewer without embedding research data files."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/soa_async_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/soa_async_v1.html')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('kb_example_viewer', ROOT / 'aiq_rl/docs/kb_viewer/build_run_pages.py')
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    data = viewer.make_data(EXAMPLE)
    data['home'] = ''
    data.pop('settingURL', None)
    for report in data.get('reports', {}).values():
        identifier = report.get('job_id')
        repo = EXAMPLE / 'jobs' / str(identifier) / 'repo'
        if repo.exists():
            report['files'] = {name:{'url':str(Path('study-files') / identifier / name)}
                               for name in ['README.md', 'analysis.json', 'release_metadata_changes.json']
                               if (repo / name).exists()}
    data['summary'] = 'Saved prototype archive: 450 KB / 450 noKB training solves; no final benchmark test. Research source and data are stored separately.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    viewer.render(data, args.output)
    for identifier in ['Jfc41e23dc91c', 'Jfab91d105641']:
        repo = EXAMPLE / 'jobs' / identifier / 'repo'
        destination = args.output.parent / 'study-files' / identifier
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['README.md', 'analysis.json', 'release_metadata_changes.json']:
            shutil.copy2(repo / name, destination / name)
    print(json.dumps({'output':str(args.output), 'bytes':args.output.stat().st_size,
                      'epochs':len(data['epochs']), 'snapshots':len(data['snaps']),
                      'report_versions':len(data.get('reports',{}))}))


if __name__ == '__main__':
    main()

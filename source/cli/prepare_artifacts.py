#!/usr/bin/env python3
"""Verify and unpack the bundled ontology and retrieval artifacts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def verify_file(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path.name}')


def prepare(artifact_dir, lake_dir, force=False):
    import lancedb
    import pyarrow.parquet as pq

    manifest = json.loads((artifact_dir / 'manifest.json').read_text())
    files = manifest['files']
    for entry in files:
        verify_file(artifact_dir / entry['path'], entry['sha256'])

    lake_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lake_dir / 'lancedb'))
    existing = set(db.table_names())
    ontology_files = [e for e in files if e['kind'] == 'ontology']
    names = {e['table'] for e in ontology_files}
    present = existing & names
    if present and present != names and not force:
        raise ValueError('Partial ontology tables exist. Use a new --lake-dir or --force to restore all three.')
    for entry in [e for e in files if e['kind'] in ('ontology', 'profile_table')]:
        name = entry['table']
        if name in existing and not force:
            print(f'Keeping existing table: {name}')
            continue
        table = pq.read_table(artifact_dir / entry['path'])
        if table.num_rows != entry['rows']:
            raise ValueError(f'Unexpected row count: {name}')
        db.create_table(name, table, mode='overwrite' if force else 'create')
        print(f'Imported {name}: {table.num_rows} rows')

    for entry in files:
        if entry['kind'] != 'query_analysis':
            continue
        destination = lake_dir / entry['destination']
        if destination.exists() and not force:
            print(f'Keeping existing file: {destination.relative_to(lake_dir)}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                with gzip.open(artifact_dir / entry['path'], 'rb') as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(destination)
        print(f'Unpacked {destination.relative_to(lake_dir)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=PROJECT_ROOT / 'data' / 'artifacts')
    parser.add_argument('--lake-dir', type=Path, default=PROJECT_ROOT / 'data' / 'lake')
    parser.add_argument('--force', action='store_true', help='Replace existing ontology tables and cached artifact files')
    args = parser.parse_args()
    prepare(args.artifact_dir.resolve(), args.lake_dir.resolve(), args.force)


if __name__ == '__main__':
    main()

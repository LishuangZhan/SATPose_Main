"""Verify Release downloads and install without overwriting different local files."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, default=Path('.artifacts'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'docs/reference/asset-manifest.json').read_text())
    records = {entry['path']: entry for entry in manifest['contents']}
    for asset in manifest['files']:
        source = args.asset_dir / asset['file']
        if not source.is_file() or source.stat().st_size != asset['size'] or digest(source) != asset['sha256']:
            raise SystemExit(f'Missing, incomplete, or mismatched download: {source}')
    for asset in manifest['files']:
        source = args.asset_dir / asset['file']
        if source.suffix == '.pth':
            entries = [(asset['destination'], None)]
            archive = None
        else:
            archive = zipfile.ZipFile(source)
            entries = [(name, name) for name in archive.namelist()]
        try:
            for relative, member in entries:
                if relative not in records or not relative.startswith(('dataset/PIM_sort/', 'log_all/')):
                    raise ValueError(f'Unexpected archive path: {relative}')
                expected = records[relative]
                target = repo / relative
                if target.exists():
                    if digest(target) != expected['sha256']:
                        raise SystemExit(f'Refusing to overwrite a different local file: {target}')
                    print(f'Already verified: {relative}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                with (source.open('rb') if member is None else archive.open(member)) as inp:
                    with temporary.open('xb') as out:
                        shutil.copyfileobj(inp, out)
                if temporary.stat().st_size != expected['size'] or digest(temporary) != expected['sha256']:
                    raise ValueError(f'Installed file failed verification: {relative}')
                temporary.rename(target)
                print(f'Installed and verified: {relative}')
        finally:
            if archive is not None:
                archive.close()
    print('Weights and pose annotations are ready. Pressure frames must be supplied separately.')


if __name__ == '__main__':
    main()

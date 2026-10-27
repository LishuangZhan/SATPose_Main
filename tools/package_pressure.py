"""Package indexed pressure inputs on a machine with fast access to the dataset.

Produces independently extractable ZIP parts below the GitHub Release asset size
limit. Original NPZ bytes and subject/frame names are preserved. Does not upload.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.generator_ours import SUBJECT_DIRECTORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-path', type=Path, default=Path('dataset'))
    parser.add_argument('--pressure-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subjects', default='all', help='all or comma-separated subject IDs')
    parser.add_argument('--actions', default='*', help='* or comma-separated exact action names')
    args = parser.parse_args()
    with np.load(args.root_path / 'PIM_sort/PIM_id_V2.npz', allow_pickle=True) as z:
        indices = z['id_data'].item()
    subjects = sorted(indices) if args.subjects == 'all' else args.subjects.split(',')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit('Use an empty output directory to avoid mixing dataset versions.')
    archives = []
    for subject in subjects:
        part = 0
        output = None
        archive = None
        count = 0

        def finish():
            if archive is None:
                return
            archive.close()
            with output.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            archives.append({'file': output.name, 'size': output.stat().st_size,
                             'sha256': digest, 'frames': count})
            print(json.dumps(archives[-1]), flush=True)

        try:
            actions = sorted(indices[subject]) if args.actions == '*' else args.actions.split(',')
            for action in actions:
                for frame in indices[subject][action]:
                    if archive is None or archive.fp.tell() >= 1800 * 1024 * 1024:
                        finish()
                        part += 1
                        count = 0
                        output = args.output_dir / f'pressure-{subject}-{part:03d}.zip'
                        archive = zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1)
                    folder = SUBJECT_DIRECTORIES[subject]
                    name = f'{action}_{int(frame)}.npz'
                    source = args.pressure_root / folder / name
                    if source.stat().st_size > 20 * 1024 * 1024:
                        raise ValueError(f'Unexpectedly large pressure frame: {source}')
                    archive.write(source, f'dataset/PIMDataset_pro_V2/{folder}/{name}')
                    count += 1
        except BaseException:
            if archive is not None:
                archive.close()
            if output is not None:
                output.rename(output.with_suffix('.partial.zip'))
            raise
        else:
            finish()
    (args.output_dir / 'pressure-manifest.json').write_text(json.dumps({
        'subjects': subjects, 'actions': args.actions, 'archives': archives,
        'extract_into': 'SATPose_Main repository root',
    }, indent=2))


if __name__ == '__main__':
    main()

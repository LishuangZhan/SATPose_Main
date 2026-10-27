"""Check pose alignment, pressure availability, and optional checkpoint compatibility."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.generator_ours import SUBJECT_DIRECTORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-path', type=Path, default=Path('dataset'))
    parser.add_argument('--pressure-root', type=Path, default=Path('dataset/PIMDataset_pro_V2'))
    parser.add_argument('--subjects', default='cxw,szy,wym,hjw', help='Comma-separated subjects or all')
    parser.add_argument('--actions', default='*', help='Comma-separated exact action names or *')
    parser.add_argument('--full-pressure-scan', action='store_true', help='Check every referenced pressure filename')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()

    data = []
    for name, key in [('PIM_2d_V2.npz', 'positions_2d'), ('PIM_3d_V2.npz', 'positions_3d'),
                      ('PIM_id_V2.npz', 'id_data')]:
        with np.load(args.root_path / 'PIM_sort' / name, allow_pickle=True) as archive:
            data.append(archive[key].item())
    poses2d, poses3d, indices = data
    subjects = sorted(indices) if args.subjects == 'all' else args.subjects.split(',')
    if set(poses2d) != set(poses3d) or set(poses2d) != set(indices):
        raise ValueError('Subject sets differ between annotations and frame indices')
    frames = 0
    checked = 0
    missing = []
    for subject in subjects:
        actions = sorted(indices[subject]) if args.actions == '*' else args.actions.split(',')
        for action in actions:
            ids = indices[subject][action]
            p2 = poses2d[subject][action]
            p3 = poses3d[subject][action]
            if p2.shape != (len(ids), 18, 2) or p3.shape != (len(ids), 18, 3):
                raise ValueError(f'Misaligned arrays: {subject}/{action}')
            if not np.isfinite(p2).all() or not np.isfinite(p3).all():
                raise ValueError(f'Nonfinite pose values: {subject}/{action}')
            frames += len(ids)
            selected = ids if args.full_pressure_scan else ids[:1]
            for frame in selected:
                path = args.pressure_root / SUBJECT_DIRECTORIES[subject] / f'{action}_{int(frame)}.npz'
                checked += 1
                if not path.is_file():
                    if len(missing) < 10:
                        missing.append(str(path))
                    continue
                if not args.full_pressure_scan:
                    with np.load(path) as archive:
                        pressure = archive['pressure']
                        if pressure.shape != (256, 64) or not np.isfinite(pressure).all():
                            raise ValueError(f'Invalid pressure array: {path}')
    print(json.dumps({'subjects': subjects, 'aligned_frames': frames,
                      'pressure_paths_checked': checked, 'missing_examples': missing,
                      'pressure_scan': 'all referenced filenames' if args.full_pressure_scan else 'first frame per action'},
                     indent=2))
    if args.checkpoint:
        import torch
        from model.vat_pose_v2 import Model
        config = SimpleNamespace(n_joints=18, channel=256, frames=351, layers=3,
                                 d_hid=512, stride_num=[3, 9, 13], out_joints=18)
        model = Model(config).eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
        with torch.no_grad():
            outputs = model(torch.zeros(1, 351, 18, 2), torch.zeros(1, 1, 27, 120, 120))
        if not all(torch.isfinite(x).all() for x in outputs):
            raise ValueError('Nonfinite checkpoint outputs')
        print('Checkpoint strictly matches the model; CPU forward pass passed (synthetic input).')
    if missing:
        raise SystemExit('Pressure data is incomplete. Obtain the full required split before evaluation.')
    print('Selected checks passed. This does not reproduce full evaluation metrics.')


if __name__ == '__main__':
    main()

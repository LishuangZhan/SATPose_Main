import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common.generator_ours import ChunkedGenerator
from common.opt import opts


class DataPathTests(unittest.TestCase):
    def test_external_annotations_pressure_and_temporal_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / 'pose annotations'
            pressure = root / 'external pressure'
            (annotations / 'PIM_sort').mkdir(parents=True)
            (pressure / 'p6cxw').mkdir(parents=True)
            indices = np.array([7, 11, 19])
            np.savez(annotations / 'PIM_sort/PIM_id_V2.npz',
                     id_data={'cxw': {'walking(stable)': indices}})
            for i, frame in enumerate(indices):
                np.savez(pressure / 'p6cxw' / f'walking(stable)_{frame}.npz',
                         pressure=np.full((256, 64), i + 1, dtype=np.float64))
            key = ('cxw', 'walking(stable)', 0)
            poses2d = {key: np.zeros((3, 18, 2))}
            poses3d = {key: np.zeros((3, 18, 3))}
            gen = ChunkedGenerator(1, None, poses3d, poses2d, pad=175,
                                   root_path=annotations, pressure_root=pressure)
            batch = gen.get_batch(key, 0, 1, False, False)
            self.assertEqual(batch[1].shape, (1, 18, 3))
            self.assertEqual(batch[2].shape, (351, 18, 2))
            self.assertEqual(batch[6].shape, (27, 256, 64))
            np.testing.assert_array_equal(batch[6][:, 0, 0], [1] * 14 + [2] + [3] * 12)

    def test_root_path_accepts_missing_trailing_separator(self):
        with patch.object(sys, 'argv', ['check', '--test', '--root_path', 'external poses',
                                      '--pressure_root', 'external pressure']):
            opt = opts().parse()
        self.assertEqual(Path(opt.root_path + 'PIM_sort'), Path('external poses/PIM_sort'))
        self.assertEqual(opt.pressure_root, 'external pressure')


if __name__ == '__main__':
    unittest.main()

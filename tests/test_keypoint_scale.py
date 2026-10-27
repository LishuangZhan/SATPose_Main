import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common.load_data_vat_ours_p0 import Fusion
from common.opt import opts
from common.vat_dataset import VatDataset


class KeypointScaleTests(unittest.TestCase):
    def test_default_and_archived_scaling_through_data_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'PIM_sort').mkdir()
            # Image corners and center give known normalized coordinates.
            pixels = np.tile([[0., 0.], [640., 360.], [1280., 720.]], (2, 6, 1))
            np.savez(root / 'PIM_sort/PIM_2d_V2.npz', positions_2d={'cxw': {'walking(stable)': pixels}})
            np.savez(root / 'PIM_sort/PIM_3d_V2.npz', positions_3d={'cxw': {'walking(stable)': np.zeros((2, 18, 3))}})
            np.savez(root / 'PIM_sort/PIM_id_V2.npz', id_data={'cxw': {'walking(stable)': np.arange(2)}})
            normalized = np.tile([[-1., -0.5625], [0., 0.], [1., 0.5625]], (2, 6, 1))
            for scale in [None, 1.075, 0.8]:
                with self.subTest(scale=scale):
                    argv = ['check', '--test', '--root_path', str(root)]
                    if scale is not None:
                        argv += ['--keypoint_scale', str(scale)]
                    with patch.object(sys, 'argv', argv):
                        opt = opts().parse()
                    opt.subjects_test = 'cxw'
                    dataset = VatDataset(root / 'PIM_sort/PIM_3d_V2.npz', opt)
                    fusion = Fusion(opt, dataset, opt.root_path, train=False)
                    actual = fusion.poses_test_2d[('cxw', 'walking(stable)', 0)]
                    np.testing.assert_allclose(actual, normalized / (1.0 if scale is None else scale), atol=1e-12)
                    if scale is None:
                        self.assertEqual(opt.keypoint_scale, 1.0)

    def test_invalid_scale_rejected_before_loading(self):
        for scale in ['0', '-1', 'nan', 'inf']:
            with self.subTest(scale=scale), patch.object(sys, 'argv', ['check', '--test', '--keypoint_scale', scale]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    opts().parse()


if __name__ == '__main__':
    unittest.main()

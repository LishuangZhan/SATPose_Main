import numpy as np
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset

vat_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 10, 12, 13, 10, 15, 16],
                        joints_left=[5, 6, 7, 8, 15, 16, 17],
                        joints_right=[1, 2, 3, 4, 12, 13, 14])


class VatDataset(MocapDataset):
    def __init__(self, path, opt):
        super().__init__(fps=20, skeleton=vat_skeleton)

        self.train_list = 'yet,cyf,hjw,gjb,grx,jx,wym,jsf,wyh,xns,zrn,wyf,yj,szy,zls,cxw'.split(',')
        self.test_list = 'czp,pyl,lsq,wlj'.split(',')
        # self.train_list = ['cyf', 'czp', 'xns', 'zls']
        # self.test_list = ['cxw', 'hjw']

        # self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        # for cameras in self._cameras.values():
        #     for i, cam in enumerate(cameras):
        #         cam.update(h36m_cameras_intrinsic_params[i])
        #         for k, v in cam.items():
        #             if k not in ['id', 'res_w', 'res_h']:
        #                 cam[k] = np.array(v, dtype='float32')  # change value in the dict to type float32
        #
        #         if opt.crop_uv == 0:  # camera center normalization
        #             cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
        #                 'float32')
        #             cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
        #
        #         if 'translation' in cam:
        #             cam['translation'] = cam['translation'] / 1000
        #
        #         cam['intrinsic'] = np.concatenate(
        #             (cam['focal_length'], cam['center'], cam['radial_distortion'], cam['tangential_distortion']))

        # dataset load, {'S1':{'Directions':[[[]]], 'Photo 1':[[[]]]......}, 'S2':{}}
        data = np.load(path, allow_pickle=True)['positions_3d'].item()
        self._data = {}
        # num = 0
        for subject, actions in data.items():  # subject: user, actions: Movemvents
            self._data[subject] = {}
            for action_name, positions in actions.items():  # Directions, value
                self._data[subject][action_name] = {'positions': positions}
                # num += positions.shape[0]
        # print(num)
        # exit()

    def supports_semi_supervised(self):
        return True

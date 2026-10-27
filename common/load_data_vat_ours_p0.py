import torch.utils.data as data
import numpy as np
from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_ours import ChunkedGenerator


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset  # h36m
        self.train = train  # True
        self.keypoints_name = opt.keypoints  # cpn_ft_h36m_dbb  2D keypoints
        self.root_path = root_path  # dataset/

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')  # None
        # self.action_filter = None
        self.downsample = opt.downsample  # 1
        self.subset = opt.subset  # 1
        self.stride = opt.stride
        # print(self.stride)
        # exit()
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)  # 2D keypoints
            # out_camera_params, out_poses_3d, out_poses_2d
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            # print(len(self.poses_train))
            # exit()
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            # print(len(self.poses_test))
            # exit()
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        for subject in folder_list:
            for action in dataset[subject].keys():  # action key
                anim = dataset[subject][action]  # 'positions': positions, 'cameras': self._cameras[subject]

                pos_3d = anim['positions'] / 100
                pos_3d[:, :, [1, 2]] = pos_3d[:, :, [2, 1]]

                positions_3d = []
                # for cam in anim['cameras']:
                #     pos_3d = world_to_camera(anim['positions'], R=cam['orientation'],
                #                              t=cam['translation'])  # world_to_camera: 3dpose world->camera coordinate
                # pos_3d[:, 1:] -= pos_3d[:, :1]  # align the root node (hip joint)
                # positions_3d.append(pos_3d)

                # pos_3d[:, 1:] -= pos_3d[:, :1]  # align the root node (hip joint)
                positions_3d.append(np.array(pos_3d))
                anim['positions_3d'] = positions_3d
                # print(dataset[subject][action])
                # exit()

        self.kps_left = [5, 6, 7, 8, 15, 16, 17]
        self.kps_right = [1, 2, 3, 4, 12, 13, 14]
        self.joints_left = [5, 6, 7, 8, 15, 16, 17]
        self.joints_right = [1, 2, 3, 4, 12, 13, 14]

        # load 2d keypoints
        keypoints = np.load(self.root_path + 'PIM_sort/PIM_2d_V2.npz', allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()

        # 2dpose normalization
        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    # cam = dataset.cameras()[subject][cam_idx]
                    if self.crop_uv == 0:
                        scale_factor = 1.075  # 缩放因子
                        if scale_factor <= 1:
                            # 缩放因子<1,相机前移，画面内容减少
                            kps[:, 0] = (kps[:, 0] / 1280 - (1 - scale_factor) / 2) / scale_factor * 1280
                            kps[:, 1] = (kps[:, 1] / 720 - (1 - scale_factor) / 2) / scale_factor * 720
                        else:
                            # 缩放因子>1,相机后移，画面内容增多
                            kps[:, 0] = (kps[:, 0] / 1280 + (scale_factor - 1) / 2) / scale_factor * 1280
                            kps[:, 1] = (kps[:, 1] / 720 + (scale_factor - 1) / 2) / scale_factor * 720

                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=1280, h=720)
                    keypoints[subject][action][cam_idx] = kps

        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:  # S1,S5,S6,S7,S8
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = [self.keypoints[subject][action]]  # 2d keypoints
                out_poses_2d[(subject, action, 0)] = poses_2d[0]  # 2d keypoints

                # if subject in dataset.cameras():
                #     cams = dataset.cameras()[subject]
                #     assert len(cams) == len(poses_2d), 'Camera count mismatch'
                #     for i, cam in enumerate(cams):
                #         if 'intrinsic' in cam:
                #             out_camera_params[(subject, action, i)] = cam[
                #                 'intrinsic']  # {('S1', 'Directions 1', 0):[[[]]], ():[[[]]]} camera intrinsics

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, action, subject, cam_ind, input_pressure = self.generator.get_batch(seq_name, start_3d,
                                                                                                  end_3d, flip,
                                                                                                  reverse)

        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True,
                                                                      reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = float(1.0)

        # print(input_2D_update.shape)
        # exit()

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind, input_pressure, start_3d

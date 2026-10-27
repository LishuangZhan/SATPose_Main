import argparse
import os
import math
import time


class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=256, type=int)
        self.parser.add_argument('--d_hid', default=512, type=int)
        self.parser.add_argument('--dataset', type=str, default='PIM')
        self.parser.add_argument('-k', '--keypoints', default='PIM_2d_V2', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--keypoint_scale', type=float, default=1.0,
                                 help='2D keypoint scale: 1.0 for unperturbed input; 1.075 for the archived experiment')
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('--pressure_root', type=str, default='dataset/PIMDataset_pro_V2',
                                 help='Root containing pressure subject folders such as p6cxw')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        # self.parser.add_argument('-a', '--actions',
        #                          default='walking(stable),walking(round),waist turning,deep squat,twisting,arm-waving,hurdle step,standing on toes,side-walking,bending,hand movements,posing,taking photo,waiting',
        #                          type=str)
        # self.parser.add_argument('-a', '--actions', default='walking(stable),walking(round)', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', default=1)
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=2000)
        self.parser.add_argument('--warmup_epochs', type=int, default=3)
        self.parser.add_argument('--batch_size', type=int, default=64, help='can be changed depending on your machine')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, default=0.0)
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=351)  # 351, 27
        self.parser.add_argument('--pad', type=int, default=175)  # 175, 13
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--refine_reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='checkpoint')
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--previous_ckpt', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=18)
        self.parser.add_argument('--out_joints', type=int, default=18)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default=math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('-previous_refine_name', type=str, default='')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.root_path = os.path.join(os.path.expanduser(self.opt.root_path), '')
        self.opt.pressure_root = os.path.expanduser(self.opt.pressure_root)
        if not math.isfinite(self.opt.keypoint_scale) or self.opt.keypoint_scale <= 0:
            self.parser.error('--keypoint_scale must be a finite positive number')

        if self.opt.test:
            self.opt.train = 0

        self.opt.pad = (self.opt.frames - 1) // 2

        stride_num = {
            '27': [3, 3, 3],
            '351': [3, 9, 13],
        }

        if str(self.opt.frames) in stride_num:
            self.opt.stride_num = stride_num[str(self.opt.frames)]
        else:
            self.opt.stride_num = None
            print('no stride_num')
            exit()

        # self.opt.subjects_train = 'yet,cyf,hjw,gjb,grx,jx,wym,jsf,wyh,xns,zrn,wyf,yj,szy,zls,cxw'
        # self.opt.subjects_test = 'czp,pyl,lsq,wlj'
        self.opt.subjects_train = 'cyf,czp,gjb,grx,jx,lsq,pyl,wlj,wyh,xns,yj,zls,zrn,wyf,jsf,yet'
        self.opt.subjects_test = 'cxw,szy,wym,hjw'
        # self.opt.subjects_test = 'cxw'

        if self.opt.train:
            # logtime = time.strftime('%m%d_%H%M_%S_')
            # self.opt.checkpoint = 'log_all' + '/' + self.opt.checkpoint + '/' + logtime + '%d' % (
            #             self.opt.pad * 2 + 1) + '%s' % ('' if self.opt.refine else '_no')

            self.opt.checkpoint = 'log_all' + '/' + self.opt.checkpoint

            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

        if self.opt.train:
            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt) if not name.startswith('_'))

            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        return self.opt

import csv
import glob
import random
import logging
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_vat_ours_p0 import Fusion
from common.vat_dataset import VatDataset
from model.block.refine import refine
from model.vat_pose_v2 import Model
import os
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
opt = opts().parse()
opt.data_augmentation = False
opt.test_augmentation = False


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


class AddNoise(object):
    def __init__(self, noise_factor=0.05, seed=None):
        self.noise_factor = noise_factor
        self.seed = seed

    def __call__(self, sample):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        noise = torch.randn_like(sample) * self.noise_factor
        return sample + noise


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    pressure_transform = transforms.Resize([120, 120])

    if split == 'train':
        model_trans.train()
        model_refine.train()
    else:
        model_trans.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    action_error_sum_refine = define_error_list(actions)

    joint_error_sum = [0] * opt.n_joints
    joint_error_dimension = {'x': [0] * opt.n_joints, 'y': [0] * opt.n_joints, 'z': [0] * opt.n_joints}
    action_joint_error = {index: [0] * opt.n_joints for index in actions}
    action_num = {index: 0 for index in actions}

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind, input_pressure, frame_id = data
        [input_2D, gt_3D, batch_cam, scale, bb_box, input_pressure] = get_varialbe(split,
                                                                                   [input_2D, gt_3D, batch_cam, scale,
                                                                                    bb_box, input_pressure])

        # 将所有压力矩阵向左平移,即压力垫后移 4格 50.5 8格51.1 16格51.6
        # input_pressure[:, :, :, :-16] = input_pressure[:, :, :, 16:]
        # input_pressure[:, :, :, -16:] = 0
        # 将所有压力矩阵向右平移,即压力垫前移 4格 50.6 8格50.9 16格51.6
        # input_pressure[:, :, :, 16:] = input_pressure[:, :, :, :-16]
        # input_pressure[:, :, :, :16] = 0
        # 将所有压力矩阵向上平移,即压力垫右移 4格 50.5 8格51.1 16格51.6
        # input_pressure[:, :, :-4, :] = input_pressure[:, :, 4:, :]
        # input_pressure[:, :, -4:, :] = 0
        # 将所有压力矩阵向下平移,即压力垫左移 4格 50.5 8格51.1 16格51.6
        # input_pressure[:, :, 4:, :] = input_pressure[:, :, :-4, :]
        # input_pressure[:, :, :4, :] = 0

        input_pressure = pressure_transform(input_pressure).unsqueeze(1)
        # input_pressure = torch.where(input_pressure > 30, torch.tensor(100), torch.tensor(0))
        input_pressure = input_pressure / 100.0
        # input_pressure = torch.zeros_like(input_pressure)
        # input_pressure = torch.randn_like(input_pressure)

        # transform = transforms.Compose([AddNoise(seed=None)])
        # input_pressure = transform(input_pressure)

        # if split == 'train':
        #     output_3D, output_3D_VTE = model_trans(input_2D, input_pressure)
        # else:
        #     input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, input_pressure, model_trans)
        output_3D, output_3D_VTE = model_trans(input_2D, input_pressure)

        out_target = gt_3D.clone()
        # out_target[:, :, 0] = 0

        # out_target: 3d gt of multi frames; out_target_single: 3d gt of middle frame
        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.refine:
            pred_uv = input_2D[:, opt.pad, :, :].unsqueeze(1)
            uvd = torch.cat((pred_uv, output_3D[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            # xyz[:, :, 0, :] = 0
            output_3D = model_refine(output_3D, xyz)

        if split == 'train':
            if opt.refine:
                loss = mpjpe_cal(output_3D, out_target_single)
            else:
                loss = mpjpe_cal(output_3D_VTE, out_target) + mpjpe_cal(output_3D, out_target_single)
            # loss = mpjpe_cal(output_3D, out_target_single)

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif split == 'test':
            # output_3D[:, :, 0, :] = 0
            # 正常情况使用
            # action_error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num = test_calculation_test_detail(
            #     output_3D, out_target_single, action, action_error_sum, opt.dataset, subject,
            #     joint_error_sum, joint_error_dimension, action_joint_error, action_num, frame_id)

            # 测试中间四块压力垫数据时使用,因为要删除足迹不在中间四块压力垫的数据所以会出现batch内没有数据而返回None的情况
            result = test_calculation_test_detail(output_3D, out_target_single, action, action_error_sum, opt.dataset,
                                                  subject, joint_error_sum, joint_error_dimension, action_joint_error,
                                                  action_num, frame_id)
            if result is not None:
                action_error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num = result
            else:
                continue

            if opt.refine:
                # output_3D[:, :, 0, :] = 0
                action_error_sum_refine, _ = test_calculation_test(output_3D, out_target, action,
                                                                   action_error_sum_refine, opt.dataset, subject)

        # if i % 200 == 0 and i != 0:
        #     print(loss_all['loss'].avg)

    joint_error_sum = [value / sum(action_num.values()) for value in joint_error_sum]
    joint_error_dimension = {index: [value / sum(action_num.values()) for value in joint_error_dimension[index]] for
                             index in joint_error_dimension.keys()}
    action_joint_error = {index: [value / action_num[index] for value in action_joint_error[index]] for index in
                          action_joint_error.keys()}

    joint_error_sum = ["{:.2f}".format(num) for num in joint_error_sum]
    joint_error_dimension = {index: ["{:.2f}".format(num) for num in joint_error_dimension[index]] for index in
                             joint_error_dimension.keys()}
    action_joint_error = {index: ["{:.2f}".format(num) for num in action_joint_error[index]] for index in
                          action_joint_error.keys()}

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        if opt.refine:
            p1, p2, p1_each, p2_each = print_error_test(opt.dataset, action_error_sum_refine, opt.train)
        else:
            p1, p2, p1_each, p2_each = print_error_test(opt.dataset, action_error_sum, opt.train)

        return p1, p2, p1_each, p2_each, joint_error_sum, joint_error_dimension, action_joint_error


if __name__ == '__main__':
    print(opt)
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'PIM_sort/PIM_3d_V2.npz'

    dataset = VatDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    # train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
    #                                                shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

    # output joint numbers
    opt.out_joints = dataset.skeleton().num_joints()

    # model initialization
    model = {}
    model['trans'] = Model(opt).cuda()
    model['refine'] = refine(opt).cuda()

    model_dict = model['trans'].state_dict()
    if opt.reload:
        # model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        # for path in model_path:
        #     if os.path.split(path)[-1][0] == 'n':
        #         no_refine_path = path
        #         break

        pre_dict = torch.load(opt.previous_ckpt)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        for path in model_path:
            if os.path.split(path)[-1][0] == 'r':
                refine_path = path
                break

        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    for epoch in range(1, opt.nepoch):
        p1, p2, p1_each, p2_each, p1_joints, joints_dims, actions_joints = val(opt, actions, test_dataloader, model)

        # with open('results/joints_dims_only2d.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(['joints'] + list(joints_dims.keys()))
        #     writer.writerows(zip(p1_joints, *joints_dims.values()))
        #
        # with open('results/actions_joints_only2d.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(actions_joints.keys())
        #     writer.writerows(zip(*actions_joints.values()))

        # with open('results/p0_stride.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(['p0'])
        #     for item_index in range(len(p1_each)):
        #         writer.writerow([p1_each[item_index]])

        print('p1: %.2f, p2: %.2f' % (p1, p2))
        print(p1_each)
        print(p2_each)
        print(p1_joints)

        exit()

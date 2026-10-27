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
from model.block.refine import refine
from model.vat_pose_resnet import Model
import os
from torchvision import transforms
import math
from common.vat_dataset import VatDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
opt = opts().parse()
opt.data_augmentation = False
opt.test_augmentation = False
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def update_learning_rate(opt, iter_step, loader_length, optimizer):

    warn_up = opt.warmup_epochs * loader_length
    max_iter = opt.nepoch * loader_length
    init_lr = opt.lr
    lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
    lr = lr * init_lr

    for g in optimizer.param_groups:
        g['lr'] = lr

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    pressure_transform = transforms.Resize([120,120])

    if split == 'train':
        model_trans.train()
        model_refine.train()
    else:
        model_trans.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    action_error_sum_refine = define_error_list(actions)
    for i, data in enumerate(tqdm(dataLoader, 0)):
        if split == 'train':
            update_learning_rate(opt, i + len(dataLoader)*(epoch-1), len(dataLoader), optimizer)

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind, input_pressure, _frame_id = data
        [input_2D, gt_3D, batch_cam, scale, bb_box, input_pressure] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box, input_pressure])
        # input_pressure = pressure_transform(input_pressure).unsqueeze(1) / 100.0
        input_pressure = input_pressure.unsqueeze(1) / 100.0
        # print(input_pressure.shape, input_pressure)
        # exit()
        # print(gt_3D.shape, input_2D.shape, input_pressure.shape)
        
        # exit()
        # if split == 'train':
        #     output_3D, output_3D_VTE = model_trans(input_2D, input_pressure)
        # else:
        #     input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, input_pressure, model_trans)
        output_3D, output_3D_VTE = model_trans(input_2D, input_pressure)

        out_target = gt_3D.clone()
        # out_target[:, :, 0] = 0

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
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

            if opt.refine:
                # output_3D[:, :, 0, :] = 0
                action_error_sum_refine = test_calculation(output_3D, out_target, action, action_error_sum_refine,
                                                           opt.dataset, subject)
        if i % 500 == 0 and i != 0:
            print(loss_all['loss'].avg, optimizer_all.state_dict()['param_groups'][0]['lr'])

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        if opt.refine:
            p1, p2 = print_error(opt.dataset, action_error_sum_refine, opt.train)
        else:
            p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)

        return p1, p2


def input_augmentation(input_2D, input_pressure, model_trans):
    joints_left = [5, 6, 7, 8, 15, 16, 17]
    joints_right = [1, 2, 3, 4, 12, 13, 14]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip, input_pressure)
    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip, input_pressure)

    output_3D_flip_VTE[:, :, :, 0] *= -1
    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE


if __name__ == '__main__':
    print(opt)
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'PIM_sort/PIM_3d_V2.npz'
    # dataset_path = root_path + 'PIM_3d_cut_V2.npz'

    dataset = VatDataset(dataset_path, opt)
    actions = define_actions(opt.actions)   # '*'

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

    # 设置模型的输出关节数
    opt.out_joints = dataset.skeleton().num_joints()

    # 初始化模型
    model = {}
    model['trans'] = Model(opt).cuda()
    model['refine'] = refine(opt).cuda()

    model_dict = model['trans'].state_dict()
    if opt.reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        for path in model_path:
            if os.path.split(path)[-1][0] == 'n':
                no_refine_path = path
                break

        pre_dict = torch.load(no_refine_path)
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
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True, weight_decay=opt.weight_decay)
    for epoch in range(1, opt.nepoch):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)

        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train and not opt.refine:
            save_model_epoch(opt.checkpoint, epoch, model['trans'])

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['trans'], 'no_refine')

            if opt.refine:
                opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                      p1, model['refine'], 'refine')
            opt.previous_best_threshold = p1

        vis_lr = optimizer_all.state_dict()['param_groups'][0]['lr']
        if not opt.train:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, vis_lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, vis_lr, loss, p1, p2))

        # if epoch % opt.large_decay_epoch == 0:
        #     for param_group in optimizer_all.param_groups:
        #         param_group['lr'] *= opt.lr_decay_large
        #         lr *= opt.lr_decay_large
        # else:
        #     for param_group in optimizer_all.param_groups:
        #         param_group['lr'] *= opt.lr_decay
        #         lr *= opt.lr_decay



# CUDA_VISIBLE_DEVICES=4 python main_vat_no_aug.py --batch_size 200 --checkpoint no_aug --data_augmentation False --test_augmentation False

# CUDA_VISIBLE_DEVICES=1 python main_vat_no_aug_v2.py --batch_size 200 --checkpoint no_aug_warmup3_cos50 --data_augmentation False --test_augmentation False --nepoch 50 --warmup_epochs 3
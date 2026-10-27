import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum, _ = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    return error_sum


def test_calculation_test(predicted, target, action, error_sum, data_type, subject):
    error_sum, error_joints = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    return error_sum, error_joints


def test_calculation_test_detail(predicted, target, action, error_sum, data_type, subject,
                                 joint_error_sum, joint_error_dimension, action_joint_error, action_num, frame_id):
    # 测试中间四块压力垫数据时使用,因为要删除足迹不在中间四块压力垫的数据所以会出现batch内没有数据而返回None的情况
    result = mpjpe_by_action_p1_detail(
        predicted, target,
        action, error_sum,
        joint_error_sum,
        joint_error_dimension,
        action_joint_error,
        action_num,
        subject,
        frame_id)
    if result is not None:
        error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num = result
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
        return error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num
    else:
        return

    # 正常情况使用
    # error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num = mpjpe_by_action_p1_detail(
    #     predicted, target,
    #     action, error_sum,
    #     joint_error_sum,
    #     joint_error_dimension,
    #     action_joint_error,
    #     action_num,
    #     subject,
    #     frame_id)
    # error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    # return error_sum, joint_error_sum, joint_error_dimension, action_joint_error, action_num


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    # predicted = predicted[:, :, :9, :]
    # target = target[:, :, :9, :]
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    error_joints = []
    for joint_id in range(predicted.shape[2]):
        predicted_each = predicted[:, :, joint_id, :]
        target_each = target[:, :, joint_id, :]
        dist_each_frames = torch.norm(predicted_each - target_each, dim=len(target_each.shape) - 1)
        dist_each = torch.mean(dist_each_frames, dim=len(target_each.shape) - 2)
        error_joints.append(torch.mean(dist_each).item() * 1000)

    if len(set(list(action))) == 1:
        # end_index = action[0].find(' ')
        # if end_index != -1:
        #     action_name = action[0][:end_index]
        # else:
        action_name = action[0]
        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
    else:
        for i in range(num):
            # end_index = action[i].find(' ')
            # if end_index != -1:
            #     action_name = action[i][:end_index]
            # else:
            action_name = action[i]
            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

    return action_error_sum, error_joints


def mpjpe_by_action_p1_detail(predicted, target, action, action_error_sum,
                              outer_joint_error_sum, outer_joint_error_dimension, outer_action_joint_error, action_num,
                              subject, frame_id):
    # # 检查第5个点(左脚)和第9个点(右脚)的x坐标
    # x_coords_5 = target[:, 0, 4, 0]
    # x_coords_9 = target[:, 0, 8, 0]
    # # 创建一个掩码,如果左脚或右脚的x坐标小于-50或大于50(超出中间四块压力垫的范围),则为True
    # # mask = (x_coords_5 < -0.5) | (x_coords_5 > 0.5) | (x_coords_9 < -0.5) | (x_coords_9 > 0.5)
    # mask = (x_coords_5 < -0.4) | (x_coords_5 > 0.4) | (x_coords_9 < -0.4) | (x_coords_9 > 0.4)
    # # 使用这个掩码来过滤target3d和predicted3d中的数据
    # target = target[~mask]
    # predicted = predicted[~mask]

    assert predicted.shape == target.shape
    num = predicted.size(0)
    if num == 0:
        return
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    # 保存每帧的被试，动作，帧数，p0，pose3d和gt pose
    # for i_dist in range(dist.shape[0]):
    #     np.savez('satpose_test_p0/' + str(sum(action_num.values()) + i_dist + 1) + '.npz', subject=subject[i_dist],
    #              action=action[i_dist], frame=frame_id[i_dist].item(), p0=round(dist[i_dist].item() * 1000, 2),
    #              pose3d=predicted[i_dist][0].cpu(), gt=target[i_dist][0].cpu())

    error_joint_frames = []  # 一个batch内的所有预测结果，各关节误差，64*18
    error_joint = []  # 一个batch内的所有预测结果的均值，各关节误差，1*18
    error_joint_dimension = {'x': [], 'y': [], 'z': []}

    for joint_id in range(predicted.shape[2]):
        predicted_each = predicted[:, :, joint_id, :]
        target_each = target[:, :, joint_id, :]

        dist_each_frames = torch.norm(predicted_each - target_each, dim=len(target_each.shape) - 1)
        error_joint_frames.append(dist_each_frames * 1000)  # 一个batch内的所有预测结果，各关节误差，64*18
        dist_each = torch.mean(dist_each_frames, dim=len(target_each.shape) - 2)
        error_joint.append(torch.mean(dist_each).item() * 1000)  # 一个batch内的所有预测结果的均值，各关节误差，1*18

        x_error = torch.mean(torch.abs(predicted_each[:, :, 0] - target_each[:, :, 0])).item() * 1000
        y_error = torch.mean(torch.abs(predicted_each[:, :, 1] - target_each[:, :, 1])).item() * 1000
        z_error = torch.mean(torch.abs(predicted_each[:, :, 2] - target_each[:, :, 2])).item() * 1000

        error_joint_dimension['x'].append(x_error)
        error_joint_dimension['y'].append(y_error)
        error_joint_dimension['z'].append(z_error)
    error_joint_frames = torch.cat(error_joint_frames, dim=1)  # 一个batch内的所有预测结果，各关节误差，64*18

    outer_joint_error_sum = [sum(pair) for pair in zip(outer_joint_error_sum, [tmp * num for tmp in error_joint])]
    outer_joint_error_dimension = {
        index: [sum(pair) for pair in
                zip(outer_joint_error_dimension[index], [tmp * num for tmp in error_joint_dimension[index]])] for index
        in outer_joint_error_dimension.keys()}

    if len(set(list(action))) == 1:
        action_name = action[0]
        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)

        action_num[action_name] += num
        outer_action_joint_error[action_name] = [sum(pair) for pair in zip(outer_action_joint_error[action_name],
                                                                           [tmp * num for tmp in error_joint])]
    else:
        for i in range(num):
            action_name = action[i]
            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

            action_num[action_name] += 1
            outer_action_joint_error[action_name] = [sum(pair) for pair in
                                                     zip(outer_action_joint_error[action_name], error_joint_frames[i])]

    return action_error_sum, outer_joint_error_sum, outer_joint_error_dimension, outer_action_joint_error, action_num


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    # predicted = predicted[:, :, :9, :]
    # target = target[:, :, :9, :]
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        # end_index = action[0].find(' ')
        # if end_index != -1:
        #     action_name = action[0][:end_index]
        # else:
        action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            # end_index = action[i].find(' ')
            # if end_index != -1:
            #     action_name = action[i][:end_index]
            # else:
            action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions(action):
    actions = ["walking(stable)", "walking(round)", "waist turning", "deep squat", "twisting", "inline lunge",
               "arm-waving", "hurdle step", "standing on toes", "side-walking", "lying", "sit-ups", "bending",
               "sitting", "push-ups", "hand movements", "posing", "Active Straight-Leg Raise", "taking photo",
               "waiting", "Rotary Stability"]

    # actions = ["walking(stable)", "waist turning", "deep squat", "twisting", "inline lunge",
    #            "arm-waving", "hurdle step", "standing on toes", "bending",
    #            "sitting", "push-ups", "hand movements", "posing", "Active Straight-Leg Raise",
    #            "waiting", "Rotary Stability"]

    if action == "All" or action == "all" or action == '*':
        return actions
    actions = action.split(',')

    return actions


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1': AccumLoss(), 'p2': AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)
    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2, _, _ = print_error_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2


def print_error_test(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2, each_error_p1, each_error_p2 = print_error_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2, each_error_p1, each_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    mean_error_all_p1 = []
    mean_error_all_p2 = []
    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)
        mean_error_all_p1.append(mean_error_each['p1'])

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)
        mean_error_all_p2.append(mean_error_each['p2'])

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))

    mean_error_all_p1 = ["{:.2f}".format(num) for num in mean_error_all_p1]
    mean_error_all_p2 = ["{:.2f}".format(num) for num in mean_error_all_p2]

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg, mean_error_all_p1, mean_error_all_p2


def save_model(previous_name, save_dir, epoch, data_threshold, model, model_name):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)

    return previous_name


def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))

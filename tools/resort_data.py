import numpy as np
from copy import deepcopy

data2d = np.load('/data/zhoujunsheng/workspace_zls/VITPose_PoseEstimation/tools/PIM_id_save/PIM_2d_V2.npz', allow_pickle=True)
data3d = np.load('/data/zhoujunsheng/workspace_zls/VITPose_PoseEstimation/tools/PIM_id_save/PIM_3d_V2.npz', allow_pickle=True)
ids = np.load('/data/zhoujunsheng/workspace_zls/VITPose_PoseEstimation/tools/PIM_id_save/PIM_id_V2.npz', allow_pickle=True)
# data3d = np.load('../dataset/PIM_3d_cut_V2.npz', allow_pickle=True)

data2d_new = data2d['positions_2d'].item()

for key in data2d['positions_2d'].item():
    # print(key)
    for value in data2d['positions_2d'].item()[key]:
        data = data2d['positions_2d'].item()[key][value]
        ind = ids['id_data'].item()[key][value]
        sorted_indices = np.argsort(ind)
        data = data[sorted_indices]
        data2d_new[key][value] = data

        # print(data2d_new[key][value])
        # # print(value)
        # exit()

data3d_new = data3d['positions_3d'].item()

for key in data3d['positions_3d'].item():
    # print(key)
    for value in data3d['positions_3d'].item()[key]:
        data = data3d['positions_3d'].item()[key][value]
        ind = ids['id_data'].item()[key][value]
        sorted_indices = np.argsort(ind)
        data = data[sorted_indices]
        data3d_new[key][value] = data

np.savez('../dataset/PIM_sort/PIM_3d_V2.npz', positions_3d=data3d_new)
np.savez('../dataset/PIM_sort/PIM_2d_V2.npz', positions_2d=data2d_new)

# data2d = data2d
# data2d = data2d['zls']['walking(round)']
# print(data2d.max(), data2d.min())

# data3d = data3d['positions_3d'].item()
# data3d = data3d['zls']['walking(round)']
# print(data3d.max(), data3d.min())


# ids_data = ids['id_data'].item()
# ids_data = ids_data['zls']['walking(round)']
# print(ids_data)
# print(data3d)

# sorted_indices = np.argsort(ids_data)
# ids_data = ids_data[sorted_indices]
# print(ids_data)
# data3d = data3d[sorted_indices]
# print(data3d.max(), data3d.min())
# print(data3d)

import numpy as np

data2d = np.load('../dataset/PIM_2d_cut_V2.npz', allow_pickle=True)
data3d = np.load('../dataset/PIM_3d_cut_V2.npz', allow_pickle=True)

data2d = data2d['positions_2d'].item()
data2d = data2d['zls']['walking(round)']
print(data2d.max(), data2d.min())

data3d = data3d['positions_3d'].item()
data3d = data3d['zls']['waist turning']
print(data3d.max(), data3d.min())

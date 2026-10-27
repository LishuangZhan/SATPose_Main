import torch

# 创建两个相同维度的随机张量来模拟数据
data1 = torch.randn(8, 1, 18, 3)
data2 = torch.randn(8, 1, 18, 3)

# 人为设置data1中的一些x坐标超出[-50, 50]范围，以确保有数据会被删除
data1[0, 0, 4, 0] = 60   # 第1个样本的第5个点的x坐标设置为60
data1[1, 0, 8, 0] = -60  # 第2个样本的第9个点的x坐标设置为-60

# 检查data1中第5个点和第9个点的x坐标
x_coords_5 = data1[:, 0, 4, 0]  # 第5个点的x坐标
x_coords_9 = data1[:, 0, 8, 0]  # 第9个点的x坐标

# 创建一个掩码，如果第5个或第9个点的x坐标小于-50或大于50，则为True
mask = (x_coords_5 < -50) | (x_coords_5 > 50) | (x_coords_9 < -50) | (x_coords_9 > 50)

# 使用这个掩码来过滤data1和data2中的数据
filtered_data1 = data1[~mask]
filtered_data2 = data2[~mask]

# 输出过滤后的数据大小
print("Filtered data1 shape:", filtered_data1.shape)
print("Filtered data2 shape:", filtered_data2.shape)

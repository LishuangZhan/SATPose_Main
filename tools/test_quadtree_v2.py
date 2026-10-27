import numpy as np
from pyqtree import Index

pressure_matrix = np.load('tools/waiting_4.npz')['pressure']


# 定义一个简单的QuadTree处理函数
def quadtree_process(matrix, threshold):
    # 将矩阵中小于阈值的元素置为0
    matrix[matrix < threshold] = 0
    # 获取矩阵中非零元素的坐标
    non_zero_coords = np.transpose(np.nonzero(matrix))

    # 创建QuadTree对象并添加非零元素的坐标
    qt = Index(bbox=(0, 0, matrix.shape[1], matrix.shape[0]))
    for coord in non_zero_coords:
        qt.insert(coord, coord)

    # 查询四叉树，获取叶子节点
    leaves = list(qt.intersect((0, 0, matrix.shape[1], matrix.shape[0])))

    # 创建一个新的矩阵，将叶子节点的坐标对应的压力值设置为100
    result_matrix = np.zeros_like(matrix)
    for leaf in leaves:
        x, y = leaf[1], leaf[0]
        if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
            result_matrix[y, x] = 100  # 注意坐标的顺序需要调整

    return result_matrix


# 指定阈值进行四叉树处理
threshold_value = 10
result_matrix = quadtree_process(pressure_matrix, threshold_value)

# 打印结果
print("原始压力矩阵:")
print(pressure_matrix)
print("\n经过四叉树处理后的压力矩阵:")
print(result_matrix)

print(pressure_matrix.max())
print(pressure_matrix.min())
print(result_matrix.max())
print(result_matrix.min())

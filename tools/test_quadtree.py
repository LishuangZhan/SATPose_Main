import numpy as np
from matplotlib import pyplot as plt


class QuadTreeNode:
    def __init__(self, pressure_matrix, x, y, width, height, threshold):
        self.pressure_matrix = pressure_matrix
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.threshold = threshold
        self.is_leaf = True
        self.average_pressure = None
        self.children = [None] * 4
        self.average_pressure = 0

    def split(self):
        if self.width > 1 and self.height > 1:
            half_width = self.width // 2
            half_height = self.height // 2

            self.children[0] = QuadTreeNode(self.pressure_matrix, self.x, self.y, half_width, half_height,
                                            self.threshold)
            self.children[1] = QuadTreeNode(self.pressure_matrix, self.x + half_width, self.y, half_width, half_height,
                                            self.threshold)
            self.children[2] = QuadTreeNode(self.pressure_matrix, self.x, self.y + half_height, half_width, half_height,
                                            self.threshold)
            self.children[3] = QuadTreeNode(self.pressure_matrix, self.x + half_width, self.y + half_height, half_width,
                                            half_height, self.threshold)

            for child in self.children:
                child.split()

            self.is_leaf = False

    def compute_average_pressure(self):
        total_pressure = 0
        total_cells = max(1, self.width * self.height)

        for y in range(self.y, self.y + self.height):
            for x in range(self.x, self.x + self.width):
                total_pressure += self.pressure_matrix[y][x]

        self.average_pressure = total_pressure / total_cells

    def process(self):
        if self.is_leaf:
            self.compute_average_pressure()
            if self.width > 1 and self.height > 1:
                if self.pressure_difference() > self.threshold:
                    self.split()
        else:
            for child in self.children:
                child.process()

    def pressure_difference(self):
        max_diff = 0

        for y in range(self.y, self.y + self.height):
            for x in range(self.x, self.x + self.width):
                diff = abs(self.pressure_matrix[y][x] - self.average_pressure)
                max_diff = max(max_diff, diff)

        return max_diff

    def get_result_matrix(self):
        result_matrix = np.zeros((self.height, self.width))

        if self.is_leaf:
            result_matrix[self.y:self.y + self.height, self.x:self.x + self.width] = self.average_pressure
        else:
            for child in self.children:
                child_result = child.get_result_matrix()
                child_height, child_width = child_result.shape

                if child_height > 0 and child_width > 0:
                    # result_matrix[:child_height, :child_width] += child_result
                    result_matrix[child.y:min(child.y + child_height, child.y + self.height),
                    child.x:min(child.x + child_width, child.x + self.width)] += \
                        child_result[:min(child_height, self.height - child.y), :min(child_width, self.width - child.x)]

        return result_matrix

    def visualize(self, ax):
        if self.is_leaf:
            ax.add_patch(plt.Rectangle((self.x, self.y), self.width, self.height, fill=None, edgecolor='r'))
        else:
            for child in self.children:
                child.visualize(ax)


def visualize_matrix(matrix, title, ax):
    ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect='auto')
    ax.set_title(title)


def visualize_quadtree(root, ax):
    root.visualize(ax)
    ax.set_xlim(0, root.pressure_matrix.shape[1])
    ax.set_ylim(0, root.pressure_matrix.shape[0])
    ax.set_aspect('equal', 'box')


def quadtree_pressure_processing(pressure_matrix, threshold):
    height, width = pressure_matrix.shape

    root = QuadTreeNode(pressure_matrix, 0, 0, width, height, threshold)
    root.process()

    result_matrix = root.get_result_matrix()

    return result_matrix, root


# 示例
pressure_matrix = np.load('waiting_4.npz')['pressure']
print(pressure_matrix.max())
print(pressure_matrix.min())
threshold = 70
result, root = quadtree_pressure_processing(pressure_matrix, threshold)
print(result.max())
print(result.min())

# # 创建一个包含三个子图的图形
# fig, axs = plt.subplots(1, 3, figsize=(6, 8))
#
# # 可视化输入矩阵
# visualize_matrix(pressure_matrix, 'Input Pressure Matrix', axs[0])
#
# # 可视化四叉树边界框
# visualize_quadtree(root, axs[1])
#
# # 可视化输出矩阵
# visualize_matrix(result, 'Output Pressure Matrix', axs[2])
#
# plt.show()

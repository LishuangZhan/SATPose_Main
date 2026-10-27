# SATPose_Main

基于 PyTorch 的人体姿态估计项目，包含使用二维关键点和压力数据的模型、训练与评估代码。

## 目录

- `main_vat.py`、`main_vat_p0.py`、`main_p0_resnet_v2.py`：训练入口。
- `eval_vat.py`、`eval_vat_p0.py`、`eval_vat_p0_resnet.py`：评估入口。
- `common/`：数据加载、相机与骨架处理、运行参数和公共工具。
- `model/`：模型及网络模块。
- `tools/`：数据处理与实验工具。

## 运行准备

运行参数定义在 `common/opt.py`。默认数据根目录为 `dataset/`，PIM 数据放在 `dataset/PIM_sort/` 下。训练入口使用 CUDA，请在配置好相应依赖和 GPU 的环境中运行。

原项目未提供依赖版本清单；本次上传保留原始 Python 源码，未验证完整训练或评估流程。

## 未纳入版本管理的内容

数据集、模型权重、训练日志、评估结果和测试生成文件需要单独准备，不包含在仓库中。`.gitignore` 已排除 `dataset/`、`log_all/`、`results/`、`satpose_test_p0/`、NumPy 数据文件、模型权重及本地缓存。

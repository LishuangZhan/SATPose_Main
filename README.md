# SATPose_Main

基于 PyTorch 的人体姿态估计代码，使用二维关键点和压力数据。仓库保存训练、评估和数据处理源码；大文件通过 [reproduction-v1 Release](https://github.com/LishuangZhan/SATPose_Main/releases/tag/reproduction-v1) 提供。

**复现状态：两个历史命令引用的权重、完整姿态标注和帧索引已整理；完整压力数据尚未发布，因此仅下载仓库及当前 Release 还不能完成评估或训练。当前版本也尚未重新验证论文指标。** 仓库当前为私有，复现者需要仓库访问权限。

## 1. 环境

推荐在有 NVIDIA GPU 的环境中运行完整训练和评估，现有入口使用 CUDA。`requirements.txt` 是本次检查过的兼容依赖组合，不是原始训练环境导出。历史记录只写了 `pycode/py37/fms` 环境名，无法据此恢复准确版本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。PyTorch 的 CUDA 安装包需要与机器驱动适配；这里固定的 PyTorch/torchvision 版本为 `2.2.2/0.17.2`。验证工具使用 Python 3.11+ 的标准库，建议统一使用 Python 3.12。`tools/` 下独立可视化实验可能还需 `matplotlib`、`pyqtree`，不影响主训练与评估依赖。

## 2. 下载权重和姿态数据

Release 包含两个权重（每个 111,312,535 字节）和 `pim-pose-annotations.zip`。权重超过普通 Git 的单文件限制，所以没有放进 Git 历史。

登录有访问权限的 GitHub 账号后，可以在仓库根目录执行：

```bash
gh release download reproduction-v1 --repo LishuangZhan/SATPose_Main --dir .artifacts --pattern '*.pth' --pattern 'pim-pose-annotations.zip'
python tools/install_reproduction_assets.py --asset-dir .artifacts
```

也可以在 Release 页面下载这三个文件，放入 `.artifacts/` 后运行安装脚本。脚本按仓库内的 SHA-256 清单校验下载文件，再安装到原命令使用的路径，不覆盖内容不同的本地文件。

```text
SATPose_Main/
├── dataset/PIM_sort/
│   ├── PIM_2d_V2.npz       # positions_2d：二维关键点
│   ├── PIM_3d_V2.npz       # positions_3d：三维标注
│   └── PIM_id_V2.npz       # id_data：对齐到压力数据的原始帧号
└── log_all/
    ├── 2d_p0_5823.pth
    └── no_refine_40_5453.pth
```

上述三个 NPZ 包含 Python 对象字典，由原项目生成；只加载可信来源的文件。

## 3. 准备压力数据（当前缺少的必要部分）

压力数据不在原 `SATPose_Main` 目录内。原代码从 `D:/pythonProject/DataProcessing/PIMDataset_pro_V2/` 读取；归档硬盘中对应的是 `Projects/pythonProject/DataProcessing/PIMDataset_pro_V2/`。

需要的布局为：

```text
PIMDataset_pro_V2/
├── p6cxw/walking(stable)_0.npz
├── p13szy/...
├── p4wym/...
├── p2hjw/...
└── 其他训练被试目录/...
```

每个帧文件至少包含形状为 `(256, 64)` 的 `pressure` 数组。文件名使用 `PIM_id_V2.npz` 中的帧号，不应重新编号。数据读取时取 27 帧压力和 351 帧二维关键点，边界按原逻辑补齐。

| 范围 | 被试 | 对齐帧数 |
|---|---|---:|
| 完整数据 | 20 名被试，每人 21 个动作 | 779,769 |
| 默认测试集全部动作 | `cxw,szy,wym,hjw` | 154,913 |
| 历史最后一条命令的 16 个动作 | 同上 | 119,787 |

按读到的 132,550 字节/帧样本估算，完整原始帧文件约 96.26 GiB；这是估算，不是完整目录实测大小。当前归档盘读取单个小文件曾耗时 30–50 秒，因此尚未完整打包或发布压力数据。

取得数据后，可以保留在外部目录，通过 `--pressure_root` 指定。姿态标注及索引的位置统一由 `--root_path` 控制，两者都支持含空格的路径。

先检查数据：

```bash
python tools/check_reproduction.py --pressure-root '/path/to/PIMDataset_pro_V2' --checkpoint log_all/no_refine_40_5453.pth
```

默认检查测试集姿态数据对齐、每个动作首帧的压力内容和可选的权重兼容性。加 `--full-pressure-scan` 会检查所选范围所有引用的压力文件是否存在；加 `--subjects all` 检查训练和测试全部被试。通过抽样检查不代表完整数据已经就绪。

原数据持有者可在能快速访问硬盘的机器上生成压缩包：

```bash
python tools/package_pressure.py --pressure-root 'D:/pythonProject/DataProcessing/PIMDataset_pro_V2' --output-dir pressure-packages
```

只打包测试数据可加 `--subjects cxw,szy,wym,hjw`。脚本按帧索引选文件、保留原始 NPZ 字节，生成按被试分组的 ZIP 分片及校验清单；解压到仓库根目录后位于 `dataset/PIMDataset_pro_V2/`。脚本不会自动上传。

## 4. 历史评估命令

原始记录保存在 [Run Code.txt](docs/reference/Run%20Code.txt)，最后一次保存的参数见 [recorded_eval_options.txt](docs/reference/recorded_eval_options.txt)。后者明确引用 `no_refine_40_5453.pth`、batch size 8 和下面的 16 个动作。

取得完整压力测试数据后，在仓库根目录运行：

```bash
python eval_vat_p0.py --test --reload --previous_ckpt log_all/no_refine_40_5453.pth --batch_size 8 --pressure_root '/path/to/PIMDataset_pro_V2' --actions 'walking(stable),waist turning,deep squat,twisting,inline lunge,arm-waving,hurdle step,standing on toes,bending,sitting,push-ups,hand movements,posing,Active Straight-Leg Raise,waiting,Rotary Stability'
```

另一条历史命令引用 `2d_p0_5823.pth`，未限制动作：

```bash
python eval_vat_p0.py --test --reload --previous_ckpt log_all/2d_p0_5823.pth --pressure_root '/path/to/PIMDataset_pro_V2'
```

这里额外加了 `--test`，避免评估入口将默认 `train=1` 写成训练日志，并显示逐动作指标；`--pressure_root` 替代写死的 D 盘路径。没有根据文件名推断权重是全局最佳，也没有根据 `2d_` 前缀改变网络分支：当前 `Model` 始终使用二维关键点和压力特征，两份权重均能严格加载。

**历史配置尚存不确定性：** `common/load_data_vat_ours_p0.py` 中 `scale_factor=1.075` 会变换二维关键点，可能属于相机位置扰动实验。本次保留该设置和现有评估计算，不擅自改为 `1.0`。如果要复现论文基准表格，需要作者确认对应权重、原始依赖版本和这项设置。权重文件名中的数字不能替代对应实验的验证结果。

## 5. 训练

历史训练命令对应：

```bash
python main_vat_p0.py --batch_size 64 --checkpoint pressure_2d_multiframe_fake_p0 --data_augmentation False --test_augmentation False --pressure_root '/path/to/PIMDataset_pro_V2'
```

P0 入口本身会把两种 augmentation 设为 `False`。实际被试划分来自 `common/opt.py` 的 `subjects_train` 和 `subjects_test`，不要用 `vat_dataset.py` 中未用于此数据划分的旧列表。完整训练还需要其余 16 名被试的压力帧。

## 6. 已验证范围和变更

- 两个权重均严格匹配模型的 277 个 state-dict 条目，并通过 CPU 合成输入前向计算，输出形状为 `(1,1,18,3)` 和 `(1,351,18,3)`。
- 全部 779,769 帧的二维关键点、三维标注和索引对齐，姿态数组未发现非有限数值。
- 原始权重和姿态数据按字节复制并进行 SHA-256 校验，详见 [asset-manifest.json](docs/reference/asset-manifest.json)。
- 修复了压力数据路径写死、帧索引忽略 `--root_path`，以及两个 P0 训练入口接收 9 项但数据加载器返回 10 项的问题；未修改模型结构、损失或指标定义。
- 回归检查：`python -m unittest discover -s tests -v`。检查记录见 [validation.json](docs/reference/validation.json)。CPU 检查不等于 CUDA 全量训练或评估成功。

原 `log_all/checkpoint/train.log` 为空，未作为结果上传。其他结果文件没有可靠的“权重—命令—指标”对应关系，未当作已验证基准发布。`satpose_test_p0/` 是生成结果，不是训练/评估的数据输入。

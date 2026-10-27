# SATPose_Main

基于 PyTorch 的人体姿态估计代码，使用二维关键点和压力数据。仓库保存训练、评估和数据处理源码；大文件通过 [reproduction-v1 Release](https://github.com/LishuangZhan/SATPose_Main/releases/tag/reproduction-v1) 提供。

**复现材料：两个历史命令引用的权重、完整姿态标注和帧索引通过 Release 提供；压力数据已在其他位置独立发布，本仓库不重复存储。** 仓库已公开，任何人都可以查看源码并下载 Release 附件。

## 1. 环境

推荐在有 NVIDIA GPU 的环境中运行完整训练和评估，现有入口使用 CUDA。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. 下载权重和姿态数据

Release 包含两个权重（每个 111,312,535 字节）和 `pim-pose-annotations.zip`。

使用已配置的 GitHub CLI，可以在仓库根目录执行以下命令；也可无需登录，直接从 Release 页面下载附件：

```bash
gh release download reproduction-v1 --repo LishuangZhan/SATPose_Main --dir .artifacts --pattern '*.pth' --pattern 'pim-pose-annotations.zip'
python tools/install_reproduction_assets.py --asset-dir .artifacts
```

也可以在 Release 页面下载这三个文件，放入 `.artifacts/` 后运行安装脚本。

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

## 3. 获取已独立发布的压力数据

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

取得数据后，可以保留在外部目录，通过 `--pressure_root` 指定。姿态标注及索引的位置统一由 `--root_path` 控制，两者都支持含空格的路径。

先检查数据：

```bash
python tools/check_reproduction.py --pressure-root '/path/to/PIMDataset_pro_V2' --checkpoint log_all/no_refine_40_5453.pth
```

默认检查测试集姿态数据对齐、每个动作首帧的压力内容和可选的权重兼容性。加 `--full-pressure-scan` 会检查所选范围所有引用的压力文件是否存在；加 `--subjects all` 检查训练和测试全部被试。

以下工具供原数据持有者按索引重新整理压缩包时使用；已取得发布数据的复现者无需执行：

```bash
python tools/package_pressure.py --pressure-root 'D:/pythonProject/DataProcessing/PIMDataset_pro_V2' --output-dir pressure-packages
```

只打包测试数据可加 `--subjects cxw,szy,wym,hjw`。脚本按帧索引选文件、保留原始 NPZ 字节，生成按被试分组的 ZIP 分片及校验清单；解压到仓库根目录后位于 `dataset/PIMDataset_pro_V2/`。

## 4. 评估命令与关键点缩放设置

原始记录保存在 [Run Code.txt](docs/reference/Run%20Code.txt)，最后一次保存的参数见 [recorded_eval_options.txt](docs/reference/recorded_eval_options.txt)。后者明确引用 `no_refine_40_5453.pth`、batch size 8 和下面的 16 个动作。

取得完整压力测试数据后，在仓库根目录运行：

```bash
python eval_vat_p0.py --test --reload --previous_ckpt log_all/no_refine_40_5453.pth --batch_size 8 --pressure_root '/path/to/PIMDataset_pro_V2' --actions 'walking(stable),waist turning,deep squat,twisting,inline lunge,arm-waving,hurdle step,standing on toes,bending,sitting,push-ups,hand movements,posing,Active Straight-Leg Raise,waiting,Rotary Stability'
```

另一条历史命令引用 `2d_p0_5823.pth`，未限制动作：

```bash
python eval_vat_p0.py --test --reload --previous_ckpt log_all/2d_p0_5823.pth --pressure_root '/path/to/PIMDataset_pro_V2'
```

与历史记录相比，这里额外加了 `--test`，避免评估入口将默认 `train=1` 写成训练日志，并显示逐动作指标；当前 `Model` 始终使用二维关键点和压力特征，两份权重均能严格加载。

## 5. 训练

历史训练入口的常规运行方式如下：

```bash
python main_vat_p0.py --batch_size 64 --checkpoint pressure_2d_multiframe_fake_p0 --data_augmentation False --test_augmentation False --pressure_root '/path/to/PIMDataset_pro_V2'
```

P0 入口本身会把两种 augmentation 设为 `False`。实际被试划分来自 `common/opt.py` 的 `subjects_train` 和 `subjects_test`，不要用 `vat_dataset.py` 中未用于此数据划分的旧列表。完整训练还需要其余 16 名被试的压力帧。

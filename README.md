| 目录或文件 | 内容 |
|---|---|
| `run.py` | 十种方法共用的命令行入口 |
| `configs/` | NanoDet结构、TiROD任务序列、自采四场景、项目模块配置 |
| `data/` | COCO契约、类别一致性检查、划分检查、回放混合批次 |
| `methods/` | 十种持续学习策略及共用蒸馏逻辑 |
| `memory/` | STM候选队列、LTM标注样本、PM模型归档、64维检索描述 |
| `model/` | 共享骨干与FPN、多个检测专家、逐帧路由 |
| `maintenance/` | 有符号软阈值、预算检查、固定验证与回滚 |
| `train_eval/` | 连续任务训练、历史任务测试、平均性能与遗忘量 |
| `visualization/` | 从真实任务矩阵生成热力图 |
| `tools/` | 十方法批量运行、图片与视频推理 |
| `docs/` | 原文件对应关系、报告对应关系、改动说明 |
| `nanodet/` | 保留的NanoDet检测器、COCO数据读取和基础组件 |

## 十种方法对应关系

| 方法 | 新策略文件 | 原入口 |
|---|---|---|
| FT 微调 | `methods/ft.py` | `TiROD.py` |
| CUMUL 累积训练 | `methods/cumul.py` | `TiROD_CUMUL.py` |
| REP 普通回放 | `methods/rep.py` | `TiROD_REP.py` |
| TREP 时序回放 | `methods/trep.py` | `TiROD_TREP.py` |
| KMREP 聚类回放 | `methods/kmrep.py` | `TiROD_KMREP_ResNet.py` |
| LREP 潜在回放 | `methods/lrep.py` | `TiROD_LREP.py` |
| LWF 输出蒸馏 | `methods/lwf.py` | `TiROD_LWF.py` |
| EWC 参数保护 | `methods/ewc.py` | `TiROD_EWC.py` |
| SID 选择性蒸馏 | `methods/sid.py` | `TiROD_SID.py` |
| LD 潜在蒸馏 | `methods/ld.py` | `TiROD_LD.py` |

LREP延续原入口中“图像回放并冻结骨干”的实际处理方式，未将其改称为已经缓存多尺度空间特征的实现。64维向量供记忆检索与去重使用，检测头仍接收空间特征。

## 环境与数据

先在目标设备准备匹配的 PyTorch / torchvision，再安装公共依赖：

```powershell
python -m pip install -r requirements.txt
python run.py --list-methods
```

新代码使用Python 3.8及以上语法及支持`weights_only=True`的PyTorch；PC与NX共用入口，NX上的PyTorch安装需与该设备的系统环境匹配；这里不替换设备现有驱动或CUDA。

数据集与ResNet50特征编码器权重不随项目生成。修改配置中的`data_root`使其指向已有数据。每个任务有独立的`train`、`val`、`test`图像目录和COCO JSON；配置里的路径均相对于配置文件解析，与启动命令所在目录无关。

TiROD的任务顺序为Domain1至Domain5、各High/Low，共10个任务。自采配置按室内、室外、低光、动态模糊四场景组织，实际路径可按已有划分修改。报告中的5415张图像和12类是项目数据描述，程序会从真实JSON读取数量和类别，不会生成对应图像或补写统计。

同一实验的所有JSON必须使用相同的完整类别表，包括暂时没有实例的类别。TiROD原表13类、自采表12类应分别运行；如需混合，需要先明确统一类别映射。数据检查会拒绝类别表不一致、无效框和同一路径图片跨训练/验证/测试复用。相邻帧及重复拷贝的内容泄漏仍应在数据划分时按采集序列处理。

## 运行方法

只显示配置与路径，或者检查实际数据：

```powershell
python run.py --config configs/tirod.yml --method REP --dry-run
python run.py --config configs/self_collected.yml --check-data
```

训练单个方法，以及按相同配置依次训练全部十种方法：

```powershell
python run.py --config configs/tirod.yml --method REP --device cuda:0
python tools/run_all.py --config configs/tirod.yml --device cuda:0
```

KMREP需要在配置中设置本地`resnet_weights`，文件为torchvision ResNet50标准权重字典；程序使用去掉分类层的特征和归一化输入，不会自动下载权重。其他方法不需要该文件。`initial_checkpoint`默认空，所有方法从相同随机种子初始化；如指定，必须与当前检测头类别数严格匹配。

项目模块组合独立运行：

```powershell
python run.py --config configs/project.yml --method REP --device cuda:0
```

该配置组合三重记忆回放、三个专家和场景路由。第一任务训练共享部分，后续冻结共享部分、只更新指定专家和路由；历史样本保留原专家标签供路由回放。微睡眠默认关闭，测定空闲时间与验证耗时后可通过`maintenance.enabled`打开。默认参数是可编辑的运行配置，并非报告实测结论。

## 输出与推理

输出目录按方法区分，每个任务保存训练CSV、最佳验证权重、最后训练状态、发布模型、各历史任务测试结果。顶层有`task_matrix.csv`、`metrics.json`、解析后的配置、检测器配置副本和运行环境记录。已有实验目录会拒绝覆盖，开始新实验时更改`output`。

验证选择模型使用已见任务的固定验证集；测试集只在任务完成后评估。每行测试全部已见任务，包括第一个任务，未来任务留空。遗忘量保留负值，表示旧任务性能提升。AP使用COCO的0–1尺度。

```powershell
python visualization/task_matrix.py outputs/tirod/baseline_REP/task_matrix.csv --output matrix.png
python tools/infer.py --config configs/tirod.yml --checkpoint outputs/tirod/baseline_REP/task_01_D1_High/model.pt --source sample.mp4 --output demo_output --device cuda:0
```

推理只读指定图片或视频，输出标注图片/视频及逐帧JSONL，包含模型版本、专家编号、检测框和预处理至后处理耗时；不打开桌面窗口。耗时不含视频解码、绘图和文件写入，不能直接作为摄像头到显示端的总时延。


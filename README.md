# LGMRec / LGMRecOpt

基于 TRAIN-only 协议的多模态推荐研究代码。LGMRecOpt 直接继承 LGMRec，默认 C0；可选增强均需在匹配协议下验证，不预先声明稳定提升。

最新候选：行为共现校正超边分配（v3），在保留 LGMRec 超图主体的基础上，尝试用 TRAIN 关系补充超边分配。
默认 `hyper_behavior_weight=0` 关闭，保留原有行为残差实验结果的可复现性。
公式、参考来源、消融和服务器运行命令见 [v3 实验说明](docs/HYPER_BEHAVIOR_V3.md)。

## 项目结构

```text
data/                   原始数据与运行产物
  baby/ sports/ clothing/
  runs/                 每次训练独立目录；legacy/ 保存迁移前日志和检查点
references/             实验记录、历史实现、发布包与迁移说明
  records/              原 recordmd/，包括 9 月 9 日、9 月 10—11 日总结
  experiments/          原 research/，完整保留历史日志、status、脚本
  legacy/v11_workspace/ 原源码、配置、测试及 archive；只读历史，不参加当前导入
  releases/             原 ZIP 发布包
scripts/                train.py / summarize.py：薄命令入口
src/lgmrec/             可安装的唯一 Python 包
  models/               原始 LGMRec；精简 LGMRecOpt
  common/               训练器与推荐模型基类
  utils/                数据加载、评估、配置、日志
  configs/              包内唯一配置来源
tests/                  当前维护的回归测试
pyproject.toml          包元数据、命令入口和 pytest 配置
requirements.txt        唯一依赖清单
VALIDATION.md           本次重组的验证范围与限制
LICENSE                当前材料的许可状态说明，不伪造上游授权
```

## 安装和训练

建议在已有可用的 PyTorch 环境执行；CUDA 版本按服务器环境安装，不盲目升级：

```bash
python -m pip install -e .
python -B scripts/train.py -d baby -g 0 --seed 999
```

默认即 C0，不再需要 `enhancement_mode=modal_item_graph`、`ablation=no_enhancement` 等历史内部开关。

```bash
# 原始 LGMRec 参照
python -B scripts/train.py -m LGMRec -d baby -g 0 --seed 999
# 显式选择保留的 v11 候选（并非默认升级）
python -B scripts/train.py -d baby -g 0 --seed 999 --variant id_cond
# 从项目外运行，显式指定数据和新输出目录
lgmrec-train -d baby --data-dir /absolute/project/data --run-dir /absolute/new_run
```

命令必须在项目根目录运行，或传绝对路径。数据目录保持原来的 `baby/baby.inter`、`image_feat.npy`、`text_feat.npy`，内容未变。配置文件从包目录读取，不再依赖当前工作目录。

每次运行自动生成唯一 `data/runs/<时间>-<variant>-seed<seed>-<随机编号>/`，含 `log/`、`checkpoints/` 和 `status.json`。指定的 `--run-dir` 已存在则拒绝运行，不覆盖历史结果。模型真实训练结束后直接返回结果写状态，不再解析 stdout 来猜测训练结果。

```bash
python -B scripts/summarize.py data/runs
python -B scripts/summarize.py data/runs --out data/runs/summary_new.json
python -B -m pytest -q
```

汇总不自行给“显著/等效/提升”的判断，也不把一个种子当成完整三种子结果；输出 JSON 必须是新文件。

## 保留与删除

- 主代码保留原始传播、sum HCL、单负样本、验证选模；C0 与可选 v11 使用同一训练管线。
- 移出主代码：旧兴趣模块、超图归一化、HCL 邻居降权、固定物品图、层间平均、逐版本复制的启动和解析脚本。旧版本完整保留在 references，不永久销毁。
- 根目录不再堆积版本 ZIP、旧测试输出、重复依赖文件和补丁说明。
- 原始 `.inter/.npy`、实验日志、检查点和研究结论未删除。历史日志里的绝对路径是当时环境记录，不批量改写。
- 不沿用旧版的 IMPLEMENTATION_VERSION 冒充完全未重构；当前版本明确为 `lgmrec-opt-clean-v1`。历史 checkpoints 使用原版本恢复，见迁移说明。
- 历史研究脚本移位后不承诺直接在新目录运行；不要把其旧命令粘到新主线。需要复现旧机制时恢复完整旧工作树。

## 实验边界

只允许 TRAIN 构图、负采样和参数学习。训练 CLI 不提供 TEST 开关，VALID 选模。最终冻结后的 TEST 单次评估应作为独立、经确认的发布流程，不在此次重构中执行。历史每轮已验证的结果不因代码整理而升级成新指标。

详细迁移与历史恢复：[references/MIGRATION.md](references/MIGRATION.md)。
# 第一阶段行为结构实验（2026-10-08）

LGMRecOpt 直接继承 LGMRec。行为分支只读取传入模型的 TRAIN 交互，
将交互二值化后按不同用户共现次数构图。保留原有视觉、文本、超图和
BPR/HCL/正则项；不新增可训练参数。SR-SNS 尚未实现，第一阶段仍使用
原有均匀单负采样。实现完成不代表已经验证推荐指标提升。

`behavior_view_mode=msca_struct` 显式开启分支；默认 `off` 保持基线。
新增配置如下，默认值仅用于兼容旧实验，不代表最优参数：

| 配置 | 默认 | 含义 |
|---|---|---|
| `behavior_eta` | 0.2 | 非负残差系数，0 用于回退校验 |
| `behavior_topk` | 10 | 每个物品最多保留的非自身邻居数 |
| `behavior_minimum` | 2 | 不同 TRAIN 用户的最低共现次数，正整数 |
| `behavior_residual_target` | both | both / item / user，控制加入哪一侧表示 |
| `behavior_graph_mode` | cooccurrence | cooccurrence / random_relabel |
| `behavior_graph_seed` | 0 | 图置乱专用种子，不消耗训练随机数流 |
| `behavior_self_loop` | true | 是否加入权重为 1 的自环 |
| `behavior_edge_weight` | binary | binary / count，入选边二值或共现计数权重 |
| `behavior_block_size` | 1024 | 分块构图行数，不改变图语义 |

共现计数按降序选择，平分时按物品 ID 排序。Top-K 是有向的，归一化
仍为原实现的行度 D^-1/2 A D^-1/2，不自动对称化。关闭自环时零度节点
采用零缩放，避免 NaN。分块计算避免一次保留完整物品共现乘积，但单块
在高密度数据上仍可能较大，可以减小 block_size。

`random_relabel` 用同一个随机排列重标记图的两端，保留完整拓扑、边数、
自环和度分布，打乱其与物品身份的对应。它不保持每个物品原来的度，
不能被表述为逐节点保度随机重连，也不能单凭此对照完全排除流行度影响。

第一步运行 A/B 三种子配对实验。下面的学习率等只是运行示例，正式实验
应填写待确认配置，对 A/B 一致使用。默认仅预览命令，增加 `--execute`
才启动训练；输出目录必须不存在，运行失败即停止，并保留已有产物。

```bash
python scripts/stage1_matrix.py --output data/runs/stage1-baby-20261008 --dataset baby --learning-rate 0.0005 --batch-size 512 --weight-decay 1e-5 --eta 0.2 --topk 10 --minimum 2
```

默认种子为 999、2026、2027。增加 `--ablations` 时加入 Item-only、User-only、
random_relabel 和 eta=0，共 18 次训练。脚本保存命令 manifest；每次运行
保存最佳 VALID checkpoint、日志，以及 status.json 中的完整 effective_config、
数据指纹和 behavior_graph 元数据。仅使用 VALID Recall@20 选模，脚本不评估 TEST。
更改 eta/K/最低共现次数需使用新的输出目录；不得将不同训练配置的历史
基线与新分支直接解释为纯模块增益。


# 第一阶段行为结构覆盖包

适用于本项目 2026-09-15 的 LGMRecOpt MSCA 行为分支代码。覆盖包只含本次
改动文件及说明，不含数据、日志、检查点或依赖安装包。覆盖前备份压缩包中
同名文件；如远程已做其他改动，请先 diff 合并。回退时恢复备份文件即可。

从项目根目录解压，使 `src/lgmrec`、`scripts`、`tests` 对齐现有目录。
使用服务器已有的 lgmrec-opt 环境，通过 `python scripts/train.py` 或下述
脚本启动，确保加载当前 src 而非旧 build/lib 或已安装的旧副本。

## 改动

- 行为结构的 eta、Top-K 和最低共现次数可调，默认保持历史二值数据语义。
- 共现矩阵分块计算，减少一次性物品乘积的临时内存；重复交互二值化，
  计数表示共同交互的不同用户数量。若数据存在重复交互，结果可能不同于旧版。
- 支持 both/item/user 残差、随机标签置乱、关闭自环、计数边权消融。
- 随机标签置乱保留拓扑和度分布，不保持每个物品的度；不是保度重连。
- status.json 增加完整生效配置与行为图指纹/参数/边数。
- 配对实验入口默认三种子，保存最佳 VALID 检查点，拒绝覆盖运行目录。
- 推理仍为原始主干表示加轻量残差，无新增神经网络参数；均匀单负采样、
  BPR、HCL 与正则项保持原逻辑。SR-SNS 留待第二阶段。

## 服务器验证与运行

先运行微型测试（使用合成数据，不读取正式数据）：

```bash
python -m pytest tests/test_msca_behavior.py tests/test_model.py tests/test_pipeline.py tests/test_artifacts.py -q
```

预览 A/B 共六次配对运行。以下数值是待确认配置示例，不是已证明最优参数。
若采用其他学习率，修改命令中的数值即可，脚本会对 A/B 一致应用。

```bash
python scripts/stage1_matrix.py --output data/runs/stage1-baby-20261008-r1 --dataset baby --learning-rate 0.0005 --batch-size 512 --weight-decay 1e-5 --eta 0.2 --topk 10 --minimum 2
```

确认后在同一命令末尾增加 `--execute` 启动。增加 `--ablations` 会包含
Item-only、User-only、random_relabel、eta=0，共18次运行。若已运行过，
改用新的输出目录；失败时保留已完成产物，单独运行 manifest 中未完成的
命令并为失败运行选择新的目录，不删除或覆盖历史记录。

汇总单次结果：

```bash
python scripts/summarize.py data/runs/stage1-baby-20261008-r1
```

正式比较需核对数据指纹和全部生效配置，并报告逐种子差值及均值/标准差。
当前包只完成实现与功能验证，未提供性能提升结论。TEST 仍只在最终冻结后
评估一次。包内 `PATCH_MANIFEST.json` 提供每个覆盖文件的 SHA256。

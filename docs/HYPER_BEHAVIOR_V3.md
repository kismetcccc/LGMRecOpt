# 行为共现校正超边分配：v3 候选实验

状态：待服务器验证，不是已经证明涨点的模型，不是独立 TRD-HRec。
LGMRecOpt 仍直接继承 LGMRec，保留其 CGE、MGE、GHE 和原有 BPR/HCL/正则损失。
本轮不实现第二阶段 SR-SNS，也不引入额外辅助损失。

## 研究依据与边界

- [LGMRec（AAAI 2024）](https://arxiv.org/abs/2312.16400)：局部表示与可学习全局超图的主体基础；[官方实现](https://github.com/georgeguo-cn/LGMRec)。
- [MMHCL（2025）](https://arxiv.org/abs/2504.16576)：用户/物品高阶结构补充一阶交互的研究依据；[官方实现](https://github.com/Xu107/MMHCL)。这里不移植其双超图或协同对比损失，不宣称复现其结果。
- [MSCA 官方实现](https://github.com/recomall/MSCA/blob/main/src/models/msca.py)：现有 TRAIN 共现结构分支的来源；来源说明继续保留在 `msca_behavior.py`。

这些参考方法不能在不同数据划分和评估协议下直接比较论文数字，也不据此认定哪个是当前 SOTA。
本轮的具体 logits 校正公式是独立实现的待验证组合，不宣称其新颖性已经完成文献验证。

与最初研究内容相比，v2 只在最终表示加入行为残差；v3 显式尝试让行为关系也参与超边分配。
这是一项模型层面的扩展，不是对 v2 既有效果的解释；论文只在消融支持后再调整。

## 公式与实现

设 `S` 为现有模块从 TRAIN 二值交互构建的稀疏、归一化共现图。
移除其自环后，按行重新归一化，得到 `P`；无邻居行使用单位行：

```text
P_ij = S_ij / sum_{k != i} S_ik   (j != i, 且行存在邻居)
P_ii = 1                         (无有效邻居)
L_m = X_m W_m
L'_m = L_m + rho * (P L_m - L_m)
L'_{u,m} = R_train L'_m
```

注意：`P` 是对现有 `S` 的重新行归一化，不是原始共现次数的直接均值。
校正后的物品和用户 logits 接入原有训练 Gumbel-Softmax / 评估确定性 Softmax，随后运行原 HGNN。
两个模态各自校正，不改变视觉—文本融合方式；只沿行为邻居的相同模态传递超边分配信号。
已有最终表示残差系数 `behavior_eta` 与新系数 `rho=hyper_behavior_weight` 独立。

- 只处理 item × hyperedge logits，不构建稠密 item × item 张量；每次前向增加两次稀疏乘法。
- 不新增神经网络参数；梯度仍回传到原来的超边投影参数。
- `rho=0` 默认完全关闭，不构建新模块、不额外消耗随机数。
- 孤立物品保持自身 logits；用户 logits 仍由 TRAIN 交互聚合。
- 图是非持久 buffer，从当前 TRAIN 重建，不从 checkpoint 覆盖。
- `status.json` 新增 `hyper_behavior_graph`：记录覆盖数、权重、图指纹和对照模式。

## 对照设计

所有组使用相同数据、种子、学习率、batch、weight decay、早停与选模指标。
除表内开关外不改变配置，不使用 TEST 调参。

| variant | eta | rho | 校正图 | 用途 |
|---|---:|---:|---|---|
| A | 无 | 0 | 无 | 原始 LGMRec |
| B | 0.2 | 0 | 无 | 当前 v2 行为残差基线 |
| hyper20 | 0.2 | 0.2 | 真实 | 主要候选 |
| hyper20_random | 0.2 | 0.2 | 重标记 | 校正关系真实性对照 |
| hyper20_only | 0 | 0.2 | 真实 | 区分超图校正与最终残差 |
| hyper10 / hyper40 | 0.2 | 0.1 / 0.4 | 真实 | 可选的小范围敏感性分析 |

随机对照只置换新模块的图节点身份，保留拓扑和度分布，但不保留每个物品的度。
原来最终残差的行为图仍为真实图，避免同时改变两条路径。
若后续改选 rho=0.1 或 0.4，必须在相同 rho 下补随机/仅超图对照，不能用 0.2 对照代替。

## 首轮服务器命令

先确认旧实验停止，再更新；复用现有环境，不需要下载新的特征。

```bash
cd ~/cf_space/LGMRecOpt
git pull --ff-only
conda activate lgmrec-opt
python -m pip install -e .
python scripts/stage1_matrix.py \
  --output data/runs/stage1-baby-hyper-v3-r1 \
  --dataset baby --seeds 999 2026 2027 \
  --learning-rate 0.0005 --batch-size 512 --weight-decay 1e-5 \
  --eta 0.2 --topk 10 --minimum 2 \
  --variants B hyper20 hyper20_random hyper20_only --execute
```

共 12 组。B 在当前代码重新跑一次用于协议/回归核对；不覆盖之前的 A/B。
需要重跑原始基线时在 `--variants` 中增加 `A`，共 15 组。
`--variants` 也可只补缺少的对照，且不能与 `--ablations` 同时使用。
输出目录必须不存在；再次运行请换 r2 等新名称。

## 判断和停止条件

首要比较 `hyper20 - B` 的配对 VALID Recall@20，同时报告 NDCG@20、Recall@50、每个种子和均值/样本标准差。
不要以最好的一个种子替代三种子结果，也不将微型测试解释为效果提升。
本轮固定配置后再跑，不根据 TEST 调节 rho。

若均值不优于 B、出现跨种子明显退化，或 NDCG@20 持续下降，先保留 B，不把新模式设为默认。
若真实校正与随机校正相近，不宣称收益来自真实共现；需要更充分的种子和机制诊断。
若超边校正有一致正向信号，再单独研究 rho 敏感性及 Sports/Clothing；所有尝试均保留，防止只报告赢家。
不承诺统计显著，不提前评估 TEST；方案冻结后再执行最终 TEST 评估。

## 本地验证记录（2026-10-08）

- 完整回归：119 项通过，包含合成数据训练、VALID 选模、日志/状态/检查点产物及防覆盖测试。
- 随后补充的 CUDA 微型训练与矩阵脚本测试：8 项通过，其中 7 项矩阵测试与完整回归重合。
- 新增 CUDA 测试验证稀疏校正、梯度和一次优化器更新；不是 GPU 全数据效果实验。
- 关闭模块的损失、梯度、初始化与 RNG 等价测试通过；开启模块的确定性评估和权重恢复通过。
- 没有运行真实数据集的新效果实验，也没有评估 TEST。合成数据指标不能用于论文性能结论。

# TRAIN 群组稀疏超图：实现与验证

日期：2026-10-15。LGMRecOpt 继续直接继承 LGMRec；本次替换可选 GHE 路径，没有建立独立模型。动机与文献依据见 [上一轮结果及方案](HYPER_CENTERED_RESULTS_NEXT_PLAN_20261015.md)。

## 本次实现

默认 `hypergraph_mode=prototype` 保留原 B。显式 `sparse_group` 启用以下流程：

1. 复用现有 TRAIN count 共现 row-TopK 支持集合，每行 j 定义超边 `e_j={j}∪N_train(j)`。默认K=10、最小共现=2；成员唯一，不足K不随机补齐，孤立物品只有自己。
2. 复用已有视觉/文本投影，以单位方向计算模态组内相关性，并加入协同表示与锚点的余弦相关性。
3. 在真实成员维度 softmax，再做物品→超边→物品的双阶段加权聚合。用户表示由 TRAIN 历史的超图物品表示取均值。

```text
f_i^m = normalize(X_i^m W_m)
q_j^m = normalize(mean_{i in e_j} f_i^m)
a_ij^m = softmax_{i in e_j}(dot(f_i^m,q_j^m) + cos(C_i,C_j))
Z_j^m = sum_{i in e_j} a_ij^m C_i
G_i^m = sum_{j: i in e_j} a_ij^m Z_j^m / sum_{j: i in e_j} a_ij^m
G_u^m = mean_{i in TRAIN(u)} G_i^m
```

这里 C 是现有 CGE 的物品表示，W_m 是现有 MGE 的投影，不创建新投影或额外损失。层数沿用 `n_hyper_layer`，同一次前向的多层传播复用由初始 C 和投影特征计算的 a。两模态支持集合相同，组内权重可以不同。

没有 Gumbel 抽样或关联 dropout；训练/推理采用同一公式。CUDA 的 index_add_ 与原有稀疏算子仍可能存在浮点累加顺序差异，不承诺跨硬件逐位复现。成员权重不是固定图：它通过当前投影和协同表示获得梯度。

## 保持与兼容边界

- CGE/MGE 结构、原评分、B 的行为残差保持；共享模态投影会接收新的超图梯度，因此不承诺训练后局部表示仍相同。
- 原 v_hyper/t_hyper 参数在稀疏模式下冻结且不参与前向；保留旧 state_dict 键和初始化随机数顺序，便于对照与载入。新层无可训练参数。
- 新图只从当前 TRAIN 重建，所有支持集合 buffer 均不持久化到 checkpoint。元数据记录实际支持指纹、TRAIN 图指纹、超边数、关联数、单节点超边数和聚合方式。
- 稀疏模式要求真实 count、binary、self-loop 行为图，并拒绝叠加原型中心化、user_mean、旧行为 logits 校正、旧超边质量平衡等无效或混杂配置。
- `hyper_num`、`keep_rate`、`user_hyper_aggregation` 是原型路径参数，在稀疏模式不控制路由；实际超边数为物品数，用户聚合固定为 TRAIN 均值，均在元数据明确记录。
- 不存在隐式 VALID/TEST 补边或负采样过滤；TEST 继续封存。

固定 H 的纯线性 HHᵀE 可化为图平滑，因此本实现不能单凭“超图”命名宣称创新。要验证的是任务表示驱动的组内选择和群组传播；`sparse_uniform` 提供相同支持下的均匀权重对照。

本次是整个 GHE 算子的替换，同时改变用户读出、随机路由和共享投影的 HCL 梯度路径。即使整体涨点，也不能单独归因于注意力；归因需要上述同支持对照，以及 HCL 开关配对实验。

## 实验组

| 矩阵名称 | 结构 | 原 HCL | 用途 |
|---|---|---|---|
| B | 原四原型 | 保留 | 同轮基准 |
| hyper_centered | 原型中心化 | 保留 | 上一轮候选复现 |
| centered_nohcl | 原型中心化 | 关闭 | 现成开关的损失因果诊断 |
| sparse_hyper | 新群组注意力超图 | 保留 | 主体替换对照 |
| sparse_nohcl | 新群组注意力超图 | 关闭 | 区分结构与辅助目标影响 |
| sparse_uniform | 新群组均匀权重超图 | 保留 | 同支持的聚合机制对照 |

关闭 HCL 只令其权重为0，BPR及原正则保留；为保留日志可比较性，仍计算原始HCL诊断值。均匀权重下，两个模态的 GHE 传播相同，局部 MGE 仍有各自模态信息；该组检验整套成员选择的作用，不解释成仅去除某一项注意力分数。

建议先验证主体，不一次启动所有消融。服务器项目根目录执行：

```bash
conda activate lgmrec-opt
python -m pip install -e .
python scripts/stage1_matrix.py \
  --output "data/runs/stage1-baby-sparse-hyper-$(date +%Y%m%d-%H%M%S)" \
  --variants B sparse_hyper sparse_nohcl \
  --seeds 999 2026 2027 \
  --learning-rate 0.002 --batch-size 512 --weight-decay 1e-5 \
  --epochs 1000 --stopping-step 20 --gpu 0 --execute --audit
```

共9次完整训练，超参数固定。只比较同轮同种子 VALID，不用旧B作跨版本选模对照。

若先单独检验上一轮的 HCL 假设，使用同样参数但改为新的输出目录和 `--variants hyper_centered centered_nohcl`，共6次训练。不要将两个候选修改同时叠入。

回传命令：

```bash
tar -czf "data/runs/sparse-hyper-results-$(date +%Y%m%d-%H%M%S).tar.gz" \
  --exclude='checkpoints' --exclude='recommendations' \
  data/runs/stage1-baby-sparse-hyper-*/
```

## 诊断和实施检查

`--audit` 对新结构报告真实的超边成员数、有效成员权重、非单节点超边的归一化熵、与均匀权重的差、训练/推理权重差，以及 BPR/HCL 梯度。它不会继续读取已停用的四原型投影来生成伪诊断；新模式 Monte Carlo 次数记为0。原型模式保留原诊断。

已覆盖的检查包括：稠密参考的双阶段前向与梯度、padding不注入无关物品、孤立节点、固定信号保持、非持久图状态、TRAIN支持集合、旧checkpoint键兼容、默认A/B对历史代码的精确输出/损失/梯度/RNG比较、稀疏模式禁止调用原型/Gumbel、GPU梯度，以及CLI产物与审计接口。

全量 `python -m pytest -q`：**223 passed**，包含 CUDA 检查。默认 A/B 与历史提交 `3e73d65` 的初始化参数、前向、损失、梯度、RNG 状态精确一致。

真实 Baby 数据短训练保存在 `data/runs/sparse-group-smoke-20261015/`，seed=999，各3轮；三组均成功结束并完成 TRAIN 审计，TEST 均未评估。选中轮次均为0基编号2。

| 组别 | VALID Recall@20 | VALID NDCG@20 |
|---|---:|---:|
| B | 0.0888750321 | 0.0390400520 |
| sparse_hyper | 0.0757700596 | 0.0329299369 |
| sparse_nohcl | 0.0891664524 | 0.0392057353 |

关闭 HCL 的新结构在这次短训练中略高于 B；保留 HCL 的新结构明显落后。这是早期训练观察，不能据此宣布收敛性能提升，也不据此调整学习率或损失权重。仍应完成三种子配对验证。

两组新结构均为7050条超边、31480个关联，2448条为单节点超边；非单节点超边的视觉/文本归一化熵分别为 `0.9395/0.9356`（保留HCL）和 `0.9765/0.9764`（关闭HCL）。训练/推理权重差为0。保留HCL时，固定审计批次的物品ID加权HCL/BPR梯度范数比约0.8523，关闭后为0。成员选择偏弱、单节点超边较多仍是需要追踪的风险；这些诊断不等于性能结论。

三组数据指纹均为 `92dc1b0fd5f60a4c5e8a01acaf9246b731dcd813a6605a89e083fd8a610f395f`，源码指纹均为 `4d3ddf3318cfba04bc94aaea14c357bb7dc1f75bbcab809ae815bb3de1c8fb3f`。环境为 Windows、PyTorch 2.6.0+cu124、RTX 4070 Laptop；与服务器先前完整结果不作直接性能比较。

## 仍需验证的风险

群组依赖共现支持，低支持物品可能只有单节点超边；语义权重可能过于均匀或与B残差重复。若新结构不优于B，应根据同支持均匀聚合和HCL消融定位原因，不通过不断增添模块掩盖问题。无论结构诊断是否改善，只有完整配对指标支持才能讨论晋升。

# LGMRecOpt 全面审查与下一步验证

日期：2026-10-14。审查基点 98adb81。LGMRecOpt 是 LGMRec 的透明扩展；本次没有建立另一套模型。

## 结论边界

当前可核验的最好 B 仍为历史 Baby VALID 三种子均值 R@20 约 .1013。collab_hcl 同轮 .101333 对 B .101300、N@20 同为 .0439，不能视为稳定提升。user_mean 服务器停止且没有回传，只记录用户观察“没有明显提升”，不生成它的精确指标。

此次首先修复评估与产物可靠性，并在明确标注的历史检查点上做 TRAIN 数值诊断。新模型候选从 B 出发，仅改变超边 logits 的计算。功能正确、缓解数值问题和稳定涨点分别验证，不能混为一个结论。

## 1. 协议、训练与复现

| 检查项 | 结论与处理 |
|---|---|
| 数据边界 | 构图及负采样仅用 TRAIN；VALID 排名屏蔽 TRAIN；最终 TEST 才屏蔽 TRAIN+VALID。当前 CLI 强制 report_test=false。此路径审查未发现标签泄漏。 |
| 提前舍入 | 原评估器先 round(...,4)，然后早停选模，真实微小改善会变成并列。已保留完整精度，日志仍格式化显示。A/B 共用新版本 full-ranking-train-only-full-precision-v2。必须重新同轮比较，不能把该变化算算法提升。 |
| 中断丢权重 | 原 Trainer 在训练全部结束后才写最优权重。已改每次 VALID 改善立即写临时文件、flush/fsync 后原子替换本轮唯一最优文件，不覆盖历史实验。中断可恢复已保存权重用于评估，但不包含 optimizer/RNG，不承诺无损续训。 |
| 非有限损失 | NaN/Inf 明确报错，不能误标记训练成功。 |
| 环境差异 | CUDA 使用 warn_only 确定性模式，不能保证所有稀疏算子复现。相同 B 的历史 seed2027 R@20 出现 .1017/.1015/.1014，说明万分位变化不够可靠。 |
| 可追溯性 | 新 status 记录 Python/PyTorch/CUDA/NumPy/SciPy、实际设备及源码 SHA。哈希范围是 src/lgmrec 的 .py/.yaml 原始字节，不含启动脚本；跨系统换行不同也可能导致哈希不同。命令由 matrix manifest 保留。 |
| 参数计数 | 旧日志误将冻结特征计入“Trainable parameters”；已改为只数 requires_grad=True，不改变训练参数。 |

## 2. 有权重支撑的模型诊断

使用 `data/runs/stage1-baby-20261008-r1/B-seed{999,2026,2027}` 的真实检查点：lr=.0005、batch=512、weight_decay=1e-5、同一 Baby 指纹。这些是历史 B，不是假称当前 .002 最优权重。诊断报告位于 `data/runs/audit-20261014/B-seed*-train.json`，记录 checkpoint/status SHA、配置和数据指纹。

只用 TRAIN 三元组计算梯度；每份报告对固定 logits 做 128 次随机分配/丢弃模拟、对同一批次做 3 次随机梯度探测。没有训练更新，没有 VALID/TEST 排名。

### 视觉超边集中和全零行

| 种子 | 视觉物品四超边 argmax 占用 | 视觉物品最大概率 >.99 | 训练 dropout 后视觉物品全零 | 视觉用户全零 |
|---|---|---:|---:|---:|
| 999 | 249 / 0 / 6801 / 0 | 99.858% | 39.061% | 49.787% |
| 2026 | 0 / 7026 / 24 / 0 | 99.986% | 47.856% | 49.991% |
| 2027 | 7050 / 0 / 0 / 0 | 100% | 28.630% | 50.037% |

文本仍使用全部四条超边，dropout 后物品全零约 6.244%。图像特征范数中位数 75.5747，文本为 1；历史两个投影参数范数没有同量级差异。可以确认这些检查点的视觉超边高度集中、部分软分配发生浮点下溢。不能由此称整个模型塌缩，也不能断言这是 R@20 平台的唯一原因。

4 条超边在 keep_rate=.5 下，即使每条关联都非零，也有 .5^4=6.25% 的行完全丢弃。视觉分配极端时，下溢使实际全零比例进一步增加。全零 anchor 对原 HCL 的贡献为常数 log(N)，这一 anchor 路径梯度为零；不代表该用户所有 BPR 路径失效。

### 损失和梯度不能混淆

历史 seed999 的加权 HCL 约 .971，BPR 约 .104，但物品 ID 上 HCL 梯度范数仅为 BPR 的 .57%–.65%。视觉投影梯度有时近零，随机探测中的梯度夹角也并不一致。故不支持凭损失数值大小削弱 HCL，更不支持未经诊断加入梯度手术或新对齐网络。接近零梯度的比例和夹角数值不应过度解读。

### 训练与推理分配仍存在差异

GumbelSoftmax(logits,tau=.2) 的随机均值一般不等于 softmax(logits/.2)。文本 argmax 随机变化约 70%，但 128 次 Monte Carlo 的物品均值误差仍大：split-half L1 约 .218，大于其均值对推理 L1 约 .177，不能把有限样本当精确期望。本轮不同时更改 Gumbel/dropout，以免无法归因。

## 3. 根据真实数据检查修正候选，而非直接发下一轮大实验

先测试 `hyper_l2`：只对超图输入做 L2 归一化，MGE 不变。在本地 RTX 4070 Laptop、seed999、lr=.002、batch512 的三轮真实 Baby 训练中，B R@20=.088824、hyper_l2=.089111。它们只是三轮检查，不可与已收敛 .1013 比较，更不可宣布涨点。

关键观察是两者视觉 argmax 都为 [0,0,7050,0]。l2 的视觉投影范数从 B 的 2.09 增至 8.68，视觉 item logits 范围中位数仍为 4.33；虽然其全零物品行从 41.87% 降至 6.26%，集中问题未消失。因此不直接把 l2 作为修复完成。

对这些固定检查点的反事实诊断（`data/runs/audit-20261014/counterfactual-common-component.json`）进一步显示：l2 视觉投影的一列与其它列夹角余弦约 -.93 至 -.95，说明模型利用共同视觉方向产生超边偏置。仅把投影列归一化，仍然同一 argmax，但软分配有变化，不能说表示完全相同。

去掉 TRAIN 活跃物品的平均单位特征方向后，在同一旧权重上的视觉占用变为 [629,1336,3150,1935]。这是定位用的反事实，不是重新训练或推荐效果。文本也有公共方向，但其投影列没有同样的反平行结构，所以不能简单宣称视觉公共方向天生是噪声。

## 4. 最终待验证候选：TRAIN 公共方向去除

配置：`hyper_input_norm=centered_l2`，矩阵名 `hyper_centered`。

```text
X_hat = row_L2_normalize(X)
L = X_hat @ W
L_centered = L - mean(L[TRAIN_active_items], axis=0)
```

等价于先从单位特征减去 TRAIN 活跃物品的平均特征，再用原投影。统计仅拟合 TRAIN 中出现的物品（当前 7047/7050），但冷物品仍可通过自己的特征计算输出。均值对投影保持梯度；不读验证/测试交互来拟合公共方向。

这是同一超边构造内的变换，无新增可训练参数、无额外损失或网络。保留 sum 用户聚合、原跨模态 HCL、Gumbel/dropout、行为 count 图、MGE 和评分。l2/cosine 仅作为本轮定位用的控制开关；最终验证不同时叠加 user_mean、collab_hcl、融合门或负采样改动。这里的 cosine 控制是超边原型余弦，与曾失败的行为邻居 cosine 排序不同。

中心化使 TRAIN 每列 logits 均值为零，排除了所有 TRAIN 物品严格由同一列取最大值的情况（除并列等退化情况），但少量极端点仍可能补偿大多数点，不能保证均衡兴趣或稳定增益。用户聚合还可能放大部分 logits，这次不同时修正。正式评估必须同时看推荐指标和训练诊断，而不是只看占用是否更平均。

### 三轮真实 Baby 检查结果

均为本地 RTX 4070 Laptop、seed999、lr=.002、batch512、weight_decay=1e-5，最佳轮索引均为 2；只跑 3 个 epoch。以下是早期功能及机制检查，**不是收敛性能比较**。

| 检查目录（data/runs 下） | 变体 | VALID R@20 | VALID N@20 | 视觉物品四超边占用 | 视觉用户 dropout 全零 |
|---|---|---:|---:|---|---:|
| audit-20261014-hyper-input-smoke | B | .0888236050 | .0390244206 | 0 / 0 / 7050 / 0 | 49.995% |
| 同上 | hyper_l2 | .0891107397 | .0391599931 | 0 / 0 / 7050 / 0 | 25.335% |
| audit-20261014-cosine-smoke | B | .0888750321 | .0390380929 | 0 / 0 / 7050 / 0 | 49.995% |
| 同上 | hyper_cosine | .0890335990 | .0391383437 | 0 / 0 / 7050 / 0 | 6.888% |
| audit-20261014-centered-smoke | hyper_centered | .0890850261 | .0392002256 | 693 / 3337 / 199 / 2821 | 6.239% |

`hyper_centered` 的视觉物品最大分配概率均值为 .4903，熵均值为 1.1535（四条超边上限 ln(4)≈1.3863）；视觉用户熵为 .7244，物品全零比例为 6.261%。固定 TRAIN 批次的三次随机探测中，视觉投影 BPR 梯度范数为 3.29e-5–8.12e-5，不再像 B 那样为零或数值上近零。它在此早期检查中缓解了集中和下溢，但是否维持到收敛、是否改善排序仍未知。

同种子 B 的两次运行也存在差异；centered 检查是在后续源码版本单独完成，不能与前两轮拼成严格配对或统计显著性结论。所有状态、日志、最优权重和 `train_audit.json` 均保留，状态含源码哈希与实际运行环境。早先主动中断的完整训练尝试另存 `audit-20261014-hyper-input-pilot/INTERRUPTED.txt`，不纳入结果。

## 5. 2026 文献复核与停止重复失败路线

- [MSCA，WWW 2026 作者实现](https://github.com/recomall/MSCA/blob/main/src/models/msca.py)支持协同视图对齐思想，但其 batch 候选、mean reduction、双侧更新与我们 collab_hcl 不同。我们的持平结果不能归咎论文，也不构成继续堆对齐的理由。
- [OrthoRec，2026 全文](https://arxiv.org/html/2609.02152v1)及[作者机构页](https://scholars.cityu.edu.hk/en/publications/beyond-modality-harmony-orthogonal-purification-and-topology-guid/)确认 ACM MM 2026 已录用；截至本报告日期 11 月会议尚未举行。它强调模态与协同冲突，但全套含正交净化、路由专家与额外损失，特征也与本项目不同，本轮不移植。只保留“应诊断模态贡献”的研究启发。
- [SG-URInit 官方预处理](https://github.com/Jinfeng-Xu/SG-URInit/blob/main/preprocessing/preprocess_agg.py)读取交互文件而未在该文件里筛 TRAIN；需要核实实际输入是否预先切分，不能据此指控论文泄漏，也不能直接用于本项目。公开实现未核实 LGMRec 适配，且当前 MGE 已历史聚合，暂不复做初始化。
- 已有 hyper v3/v4、行为 cosine、CSA/AMUR、HCL 邻居降权的失败或不稳定记录继续保留，不换名称重新堆入。

## 6. 运行及交付边界

新 `scripts/audit_model.py` 要求显式 checkpoint/status，并校验来源、冻结特征、数据指纹。仅 TRAIN 诊断，不评估排名、不更新参数。矩阵加 `--audit` 后每次 LGMRecOpt 完成训练自动导出 `train_audit.json`，回传压缩包可继续排除大检查点，同时保留可分析的证据。该工具目前只支持 LGMRecOpt；含 A 的矩阵会明确跳过 A 的诊断并继续，manifest 记录适用模型。

```bash
python scripts/stage1_matrix.py \
  --output "data/runs/stage1-baby-hyper-centered-$(date +%Y%m%d-%H%M%S)" \
  --variants B hyper_centered \
  --learning-rate 0.002 --batch-size 512 --weight-decay 1e-5 \
  --execute --audit
```

共六次完整训练，固定种子999/2026/2027；同轮 B 和候选使用新全精度选模规则。只看 VALID，TEST 封存。若只改善占用而没有一致的 R@20/N@20 增益，候选仍不晋升为最好模型。正式完整训练尚未完成，不能宣布稳定提升。

验证：全量 `pytest -q` 为 176 passed；之后针对审查发现的 A/`--audit` 兼容性修复，额外运行矩阵测试，15 passed。测试覆盖 TRAIN 中心化边界、梯度等价、默认参数/RNG/原局部分支不变、选模精度、中断及磁盘写失败时的检查点保存、诊断不更新权重及随机状态。

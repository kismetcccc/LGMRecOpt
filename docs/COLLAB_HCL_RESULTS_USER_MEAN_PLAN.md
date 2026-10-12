# 协同 HCL 结果与用户超边聚合候选

日期：2026-10-12。结果代码 d1c7e2c。LGMRecOpt 继续直接继承 LGMRec。

## 回传结果

`collab-hcl-results-20261012-115651.tar.gz` 与 `115656` 为相同包，SHA256 为 `d26122f6829a4863cc6b523696ec27722a2e0afda7c248b95300e2a321ba243f`，不能计作两次重复实验。
9 个 status 均 finished=true、returncode=0、test_evaluated=false。共同数据指纹为 `92dc1b0fd5f60a4c5e8a01acaf9246b731dcd813a6605a89e083fd8a610f395f`。同种子 B/collab_hcl 的有效配置仅输出路径及 hcl_view 不同。

| 方案 | seed999 R@20 | seed2026 R@20 | seed2027 R@20 | 平均 R@20 | 平均 N@20 | 平均 R@50 |
|---|---:|---:|---:|---:|---:|---:|
| A | .0986 | .0996 | .0995 | .099233 | .043533 | .162967 |
| B | .1013 | .1012 | .1014 | .101300 | .043900 | .163833 |
| collab_hcl | .1014 | .1013 | .1013 | .101333 | .043900 | .164500 |

均为 VALID，按 R@20 选取检查点。collab_hcl 的 R@20 差值为 +.0001/+.0001/-.0001，平均仅 +.000033，约相对 +.033%；原始输出四位小数，勿对均值多位小数作过度解释。不能认为已稳定涨点，也不证明原 HCL 有害。B 保留为下一轮主要对照，不叠加 collab_hcl。

## 下一处改动：用户 logits 从历史求和改为历史均值

依据是现有代码的数学性质，详见 [模型审查](PAPER_GROUNDED_MODEL_AUDIT_20261012.md)，不是宣称 2026 论文已经证明此改法有效。确定性分配中，原 `softmax(sum(logits)/tau)` 在平均 logits 相同的条件下具有有效温度 `tau/history_length`。MGE 已取历史均值，用户超边分配却取和。训练 Gumbel 阶段还涉及信号与噪声比例，不能直接套用同一有效温度解释。

候选 `user_hyper_aggregation=mean` 复用 TRAIN 历史度数，在用户 logits 进入 Gumbel/softmax 前取均值。原始 `sum` 为默认；物品 logits、图、HCL、HGNN、推理评分、所有训练超参数保留。只新增一个选择开关和一个聚合钩子，无参数、无新模块或辅助损失。与失败的 v4 不同：v4 修改传播后的超边质量归一化，本次修改分配前的用户历史聚合。

风险：更多历史可能本来就应增加分配置信度，均值可能抹除有用证据；没有当前 B 检查点，尚未证明真实模型存在有害饱和。因此这是可证伪的机制实验，不是已证实问题的修复。初始化合成测试仅验证数学性质，不能当作真实模型诊断。

## 验证方案

固定 lr=.002、batch=512、weight_decay=1e-5、eta=.2 及种子 999/2026/2027。本轮只跑 B/user_mean 共 6 次，已有多轮同协议 A，本轮直接隔离新增改动，减少重复基线训练。构图和优化只访问 TRAIN，VALID 选模，TEST 封存。

```bash
cd ~/cf_space/LGMRecOpt
python scripts/stage1_matrix.py --output "data/runs/stage1-baby-user-mean-$(date +%Y%m%d-%H%M%S)" --variants B user_mean --learning-rate 0.002 --batch-size 512 --weight-decay 1e-5 --execute
```

按同轮三种子差值、均值、NDCG@20 判断，不挑最好单次。若没有一致优势，不继续搜索温度或均值比例来保住该候选。日志、配置、检查点均写入新目录，不覆盖既有记录。

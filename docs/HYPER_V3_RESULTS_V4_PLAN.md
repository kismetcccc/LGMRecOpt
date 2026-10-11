# v3 结果复盘与 v4 超边聚合候选（2026-10-09）

## 2026-10-11 回传结果核对

本地 `data/runs/stage1-baby-hyper-v4-r1` 已有 9 组完成记录，均 returncode=0、test_evaluated=false，日志也标记 validation-only。数据指纹均为下文所列值；同种子 effective_config 仅产物路径与 hyper_degree_power 不同。

| seed | B R@20 | edge05 R@20 | edge10 R@20 |
|---|---:|---:|---:|
| 999 | 0.0997 | 0.0994 | 0.0996 |
| 2026 | 0.1003 | 0.0999 | 0.0999 |
| 2027 | 0.0986 | 0.0983 | 0.0985 |
| 均值 | 0.099533 | 0.099200 | 0.099333 |

两种归一化的三个种子均低于同轮 B，本轮不支持启用 v4。保留历史实现与产物，继续关闭 v3/v4，转向 README 所列行为机制消融。下文“尚未运行”是 10 月 9 日实现时的历史记录，不代表当前实验状态。

## 实际结果，而非性能承诺

来源：回传的 `data/runs/stage1-baby-hyper-v3-results.tar.gz`，解压到原始相对目录。
共 12 组正常完成；所有 `test_evaluated=false`，日志以 validation-only 完成。
数据指纹一致：`92dc1b0fd5f60a4c5e8a01acaf9246b731dcd813a6605a89e083fd8a610f395f`。
逐种子的有效配置除预定开关和产物目录外一致，没有用 TEST 选模。

以下为三种子 VALID 均值（样本标准差采用 n-1）：

| 组别 | R@20 均值 ± 标准差 | N@20 均值 | R@50 均值 |
|---|---:|---:|---:|
| B | 0.099433 ± 0.000737 | 0.043233 | 0.165833 |
| hyper20 | 0.099667 ± 0.001106 | 0.043267 | 0.166100 |
| hyper20_random | 0.099400 ± 0.000954 | 0.043167 | 0.163900 |
| hyper20_only | 0.097167 ± 0.000231 | 0.042633 | 0.164000 |

| seed | B R@20 | hyper20 R@20 | 配对差值 |
|---|---:|---:|---:|
| 999 | 0.0997 | 0.1007 | +0.0010 |
| 2026 | 0.1000 | 0.0998 | -0.0002 |
| 2027 | 0.0986 | 0.0985 | -0.0001 |

结论：主指标均值相对 +0.235%，但 2/3 种子下降；不能称为稳定提升。
真实图相对随机图的 R@20 均值差仅 0.000267，不足以建立强机制结论。
去掉最终行为残差的 hyper20_only 退化，当前不应替代 B。
真实校正图覆盖 4602/7050 个物品，2448 个无邻居物品保持原 logits；覆盖不全不等于已证明的失效原因。

本轮 B 与上一轮 B 的 R@20 相同，但某些最佳轮次/NDCG/R@50 有轻微差异。
因此只使用本轮配对对照解释 v3，不能把跨轮微小差值直接归因于算法。
日志没有足够的环境信息确定跨轮差异的具体原因。

决定：保留 v3 代码以便复现，但默认仍关闭，不扩大 v3 超参数搜索；保留 B。

## v4：超边成员质量归一化

原 HGNN 对每个超边直接汇总所有关联物品。下一轮只检验这个求和是否造成大超边贡献偏重。
这是由实现提出的假设，不是本轮日志已经验证的因果结论。

```text
H_i, H_u = 原始模型的物品/用户软关联矩阵（训练时使用原 dropout 后的矩阵）
d_e = sum_i H_i[i,e]
Z_e = (H_i^T E)_e / max(d_e, eps)^p
E_i' = H_i Z
E_u' = H_u Z
```

`p=0` 调用原 HGNN 路径，数值和随机数行为不变；`p=.5` 部分校正；`p=1` 超边均值聚合。
多层时每层使用相同关联矩阵重复这一操作。分母不 detach，梯度经过已有超边参数。
空超边聚合为零，分母夹到 dtype epsilon 避免零除。
这只是超边维度的归一化，不是完整的对称节点—超边 HGNN 归一化，也不宣称算法新颖性。
无新参数、无新损失、不改变模态融合；仍为 LGMRec 的透明扩展。

新配置 `hyper_degree_power=0.0` 默认关闭，旧实验可以继续复现。
首轮保留行为残差 eta=.2，关闭 v3 校正（rho=0），只比较 B / edge05 / edge10。
三个种子共 9 组；仅 VALID 选模；不覆盖已有实验，不在多个种子中挑选赢家。

## 服务器运行

```bash
cd ~/cf_space/LGMRecOpt
git pull --ff-only
conda activate lgmrec-opt
python -m pip install -e .
python scripts/stage1_matrix.py \
  --output data/runs/stage1-baby-hyper-v4-r1 \
  --dataset baby --seeds 999 2026 2027 \
  --learning-rate 0.0005 --batch-size 512 --weight-decay 1e-5 \
  --eta 0.2 --topk 10 --minimum 2 \
  --variants B edge05 edge10 --execute
```

观察相对 B 的配对 R@20，辅助报告 N@20 和 R@50。若两种校正均无一致改善，保留 B，不把 v4 设为默认。
若有信号，后续独立验证更多种子/数据集；不能用同批验证筛选后的最优结果宣称泛化提升。
本轮不叠加 v3、不做第二阶段负采样，避免多因素混杂。

## 回传结果（不包含检查点）

以下命令从项目根目录运行，将新压缩包直接放在 `data/runs`。
如果同名包已存在就停止，避免覆盖既有结果包。

```bash
if [ -e data/runs/stage1-baby-hyper-v4-results.tar.gz ]; then
  echo '结果包已存在，请改用新文件名'
else
  find data/runs/stage1-baby-hyper-v4-r1 -type f \
    \( -name 'status.json' -o -name 'manifest.json' -o -name '*.log' \) -print0 |
  tar --null -T - -czf data/runs/stage1-baby-hyper-v4-results.tar.gz
fi
```

检查点保留在服务器，本轮日志包与既有 v3 包均不上传 GitHub。

## 本地验证

2026-10-09：完整回归 135 项通过；补充数值梯度与新矩阵测试后，定向回归 23 项通过（与完整回归有重合）。
覆盖 CPU/CUDA 微型训练、空超边有限值、原始路径等价、checkpoint 恢复、端到端 VALID-only 产物及防覆盖。
12 组回传日志的最终 Valid 数值逐一与 status.json 一致。
尚未运行 v4 真实数据集性能实验；微型测试不代表推荐指标提升。

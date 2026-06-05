# Task 3 实验迭代说明

## 1. 实验目标与统一评估方式

本阶段的目标是完成 BraTS 多模态脑肿瘤分割。输入为 `T1`、`T1c`、`T2` 和 `FLAIR` 四个 MRI 模态，输出包括三个临床区域：全肿瘤 `WT`、肿瘤核心 `TC` 和增强肿瘤 `ET`。

所有正式结果都使用同一份按患者划分的数据集，比例为 `7:1:2`，随机种子为 `42`。训练阶段采用随机 `96 x 96 x 96` patch，最终测试阶段采用 stride 为 `48` 的 sliding-window 全体积推理，共评估 `127` 个测试病例。因此，下表中的测试结果可以直接横向比较。

需要区分两类目录：

- `smoke_*` 只用于确认模型能够正常训练和保存，不用于比较性能。
- `taf_transbts_cluster_mmap*` 是早期 correlation-guided TAF 探索，出现过后期退化和 `NaN`，只用于记录失败路径。
- `taf_transbts_cluster_a3*` 是关闭 correlation KL 梯度后的受控融合消融。最终可汇报结果来自 `taf_transbts_cluster_a3_finetune_resume_mmap`。

## 2. 正式实验结果

Dice 越高越好，HD95 越低越好。

| 模型 | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention U-Net | 0.9006 | 0.8930 | 0.8628 | 0.8854 | 6.12 | 3.35 | **2.76** |
| SwinUNETR v2 | 0.9089 | 0.8952 | 0.8598 | 0.8879 | 5.91 | 5.85 | 5.35 |
| **TransBTS baseline** | 0.9132 | **0.9022** | **0.8711** | **0.8955** | 4.91 | **3.26** | 3.62 |
| MSA-TransBTS | **0.9169** | 0.8938 | 0.8709 | 0.8939 | 4.74 | 3.84 | 3.00 |
| RAM-TransBTS | 0.9105 | 0.8982 | 0.8677 | 0.8921 | 12.39 | 3.53 | 2.96 |
| RAMS-TransBTS | 0.9148 | 0.8969 | 0.8654 | 0.8924 | **4.62** | 3.41 | 3.33 |
| RSF-TransBTS | 0.8893 | 0.8912 | 0.8610 | 0.8805 | 10.10 | 5.96 | 5.81 |
| **TAF-TransBTS A3** | 0.9160 | **0.9087** | **0.8774** | **0.9007** | 5.41 | 3.73 | 3.49 |

当前最可靠的结论是：`TAF-TransBTS A3` 已成为整体 Dice 最优模型。相比 `TransBTS baseline`，它的 Mean Dice 从 `0.8955` 提升到 `0.9007`，TC Dice 从 `0.9022` 提升到 `0.9087`，ET Dice 从 `0.8711` 提升到 `0.8774`。代价是 WT 和 TC 的 HD95 分别从 `4.91 / 3.26` 变为 `5.41 / 3.73`，说明重叠度提高的同时，部分病例的边界稳定性仍有优化空间。

## 3. 从基线到区域融合的迭代逻辑

### 3.1 先确定基础架构：从局部卷积走向全局建模

最初使用 Attention U-Net 建立可工作的 3D 分割基线，随后加入 SwinUNETR 和轻量版 TransBTS 进行比较。三者使用相同的全体积测试协议后，TransBTS 的 Mean Dice 达到 `0.8955`，相比 Attention U-Net 提高 `0.0101`，相比 SwinUNETR 提高 `0.0076`。

这一结果说明，在当前数据规模和显存条件下，CNN 编码器负责提取局部纹理、Transformer bottleneck 负责补充全局关系，是一个有效且相对轻量的折中。因此，后续优化都围绕 TransBTS 展开，而不是继续增加主干复杂度。

### 3.2 MSA：先尝试轻量的模态与空间注意力

四个 MRI 模态对不同肿瘤区域的贡献并不相同。基于这一点，`MSA-TransBTS` 在输入端增加轻量的 modality-spatial attention：先根据每个模态的全局统计量调整模态权重，再根据空间响应强调可能的病灶区域。

MSA 的 WT Dice 达到全部实验最高的 `0.9169`，WT HD95 也从 `4.91` 改善到 `4.74`。但 TC Dice 从 `0.9022` 降到 `0.8938`，Mean Dice 最终为 `0.8939`，仍比 baseline 低 `0.0016`。

这说明轻量注意力确实有价值，尤其能够改善较大范围的 WT 区域；但仅靠输入端统一加权，还不足以同时改善更细小、更难分的 TC 和 ET。

### 3.3 RAM 与 RAMS：让模态融合显式面向 WT、TC、ET

下一步将“不同区域依赖不同模态”的想法写得更明确。`RAM-TransBTS` 不再只学习一组全局模态权重，而是额外生成面向 `WT`、`TC`、`ET` 的三个融合通道，并与原始四模态共同输入主干。

RAM 的整体 Mean Dice 为 `0.8921`，未超过 baseline。病例级分析显示，它在 `127` 个测试病例中改善了 `64` 个、退化了 `63` 个，平均变化为 `-0.0034`。这不是一个完全无效的方向：例如在困难病例 `BraTS-GLI-00321-000` 上，Mean Dice 从 `0.4354` 提高到 `0.5076`。但它的稳定性不足，`BraTS-GLI-00675-001` 出现 WT 完全漏检，导致 WT HD95 明显恶化到 `12.39`。

`RAMS-TransBTS` 在 RAM 后继续加入 shared spatial attention，希望抑制不稳定的空间响应。结果 WT HD95 从 RAM 的 `12.39` 恢复到 `4.62`，并优于 baseline 的 `4.91`；但 Mean Dice 仍只有 `0.8924`。因此，空间注意力改善了边界稳定性，却没有解决 TC 和 ET 精细区域性能下降的问题。

### 3.4 RSF：更强的区域监督带来了新的泛化问题

在 RAM/RAMS 之后，`RSF-TransBTS` 将区域建模进一步前移到 feature level：

- 每个模态先经过独立的浅层 stem。
- 网络为空间位置学习不同的 WT、TC、ET 模态融合权重。
- 三个区域增加辅助分割头。
- 损失中加入区域辅助损失和 `ET <= TC <= WT` 的嵌套约束。
- 训练 patch 按 `WT:TC:ET:random = 0.35:0.25:0.25:0.15` 采样，使模型更频繁看到肿瘤区域。
- 每 `5` 个 epoch 使用 sliding-window 全体积验证选 checkpoint，减少中心 patch 验证带来的偏差。

RSF 的全体积验证 Mean Dice 在 epoch `40` 达到 `0.9212`，说明训练过程能够收敛；但最终测试 Mean Dice 只有 `0.8805`，相比 baseline 下降 `0.0150`。病例级结果中，仅 `18` 个病例改善，`109` 个病例退化。

从 worst-case 可视化可以看到，RSF 不只是边界略有偏差，而是会在远离真实肿瘤的位置生成断开的假阳性区域。例如 `BraTS-GLI-00321-000` 的 WT/TC/ET HD95 分别达到 `80.73 / 109.24 / 111.77`，`BraTS-GLI-00733-001` 也出现远距离假阳性区域。

这里可以做一个有证据支持的推断：区域平衡采样和更强的辅助监督提高了模型对病灶响应的敏感度，但随机背景 patch 只占 `15%`，再叠加空间门控后，模型对“看起来像病灶”的局部特征变得过于积极，导致断开的远端误检。代码中的 TAF 注释也记录了早期 RSF sigmoid gating 存在持续放大特征的问题。RSF 的负结果因此很有价值：下一步不应继续简单堆叠区域监督，而应优先约束融合方式的稳定性。

## 4. 验证分数为什么不能直接当作最终结论

前六组模型使用中心 patch 验证进行 checkpoint 选择，RSF 和 TAF 则改为周期性全体积验证。两种验证方式的难度和覆盖范围不同，因此 checkpoint selection score 只能用于理解各自训练过程，不能直接横向排名。

| 模型 | 最佳 epoch | Checkpoint selection score | Test Mean Dice | 差值 |
| --- | ---: | ---: | ---: | ---: |
| Attention U-Net | 82 | 0.9146 | 0.8854 | 0.0291 |
| SwinUNETR v2 | 79 | 0.9091 | 0.8879 | 0.0211 |
| TransBTS baseline | 97 | 0.9216 | 0.8955 | 0.0261 |
| MSA-TransBTS | 99 | 0.9240 | 0.8939 | 0.0302 |
| RAM-TransBTS | 99 | 0.9185 | 0.8921 | 0.0264 |
| RAMS-TransBTS | 97 | 0.9213 | 0.8924 | 0.0289 |
| RSF-TransBTS | 40 | 0.9212 | 0.8805 | 0.0407 |
| TAF-TransBTS A3 | 150 | 0.9359 | 0.9007 | 0.0351 |

另外，MSA、RAM 和 RAMS 使用 mild augmentation，而 baseline 使用更强的数据增强。因此，它们反映的是“结构调整加增强策略调整”后的结果，还不是严格的单变量消融实验。正式汇报时可以描述趋势，但不应将差值完全归因于注意力模块。

## 5. TAF-TransBTS：受控融合的最终结果

基于 RSF 的问题，下一轮实现了 `TAF-TransBTS`。这个方向的重点不是继续增加区域辅助头，而是让融合过程更加受控：

- 四个模态使用独立编码器，保留每种 MRI 的特征表达。
- 使用 softmax 归一化的 modality attention 和 spatial attention，使融合保持为模态特征的凸组合，避免持续放大。
- 使用 baseline augmentation，便于进行更干净的结构对比。

TAF 参数量约为 `11.27M`，高于 baseline 但仍低于 Attention U-Net。由于它需要四路独立编码器，训练在集群 RTX 4090 GPU 上完成。

### 5.1 correlation-guided TAF 的失败路径

早期 `taf_transbts_cluster_mmap` 在深层特征上增加跨模态 correlation loss，约束 `T1-T1c`、`T1-T2` 和 `T2-FLAIR` 的关系。但训练后期出现明显退化：中心 patch 验证 Mean Dice 降至 `0.0540`。后续 `taf_transbts_cluster_mmap_stable` 尝试增加梯度裁剪与数值保护，仍在后期出现 `NaN`。

因此，代码进一步增加了 FP32 correlation 路径、有界多项式输入、teacher KL、encoder 梯度缩放、可选 correlation 开关和非有限梯度跳过机制。对应配置为 `configs/task3_taf_transbts_cluster_a4_stable.yaml`。这套 A4 correlation-guided 配置已经具备重新验证条件，但当前尚无正式测试结果。

### 5.2 A3：关闭 correlation KL 后验证受控融合主体

为了将结构收益和 correlation 数值问题拆开，A3 关闭 correlation KL 梯度，仅保留四路独立编码器、modality attention、spatial attention 和 Transformer bottleneck。最终训练采用分阶段初始化与微调：

| 阶段 | 初始化方式 | 实际设置 | 结果 |
| --- | --- | --- | --- |
| `taf_transbts_cluster_a3_mmap` | 从早期 stable 最佳权重初始化 | `lr=2.5e-4` | 运行到 19 epoch 后中断，保留过渡 checkpoint |
| `taf_transbts_cluster_a3_mmap_continue` | 从 A3 过渡 checkpoint 初始化，重置优化器与 epoch | `lr=2.5e-4`, 100 epoch | 最佳 full-val Mean Dice `0.9290`，epoch `80` |
| `taf_transbts_cluster_a3_finetune_mmap` | 从上一阶段最佳权重初始化，重置优化器与 epoch | `lr=1e-4`, 50 epoch | 最佳 full-val Mean Dice `0.9320`，epoch `30` |
| `taf_transbts_cluster_a3_finetune_resume_mmap` | 使用 `--resume-checkpoint` 恢复优化器、调度器和 epoch 编号 | 从 epoch `31` 续跑至 `200` | 最佳 full-val Mean Dice `0.9359`，epoch `150` |

前两个后续阶段使用的是 `--init-checkpoint`，语义是“继承最佳模型权重后开始新的微调阶段”，不是无缝续跑。最后一个阶段使用 `--resume-checkpoint`，才是真正恢复优化器、调度器和 epoch 编号的连续训练。

最终阶段没有出现跳过的优化步骤，也没有非有限 loss。测试使用 epoch `150` 的 `best.ckpt`，在 `127` 个测试病例上完成 stride `48` 的 sliding-window 全体积推理：

| 指标 | TransBTS baseline | TAF-TransBTS A3 | 变化 |
| --- | ---: | ---: | ---: |
| WT Dice | 0.9132 | 0.9160 | +0.0028 |
| TC Dice | 0.9022 | 0.9087 | +0.0065 |
| ET Dice | 0.8711 | 0.8774 | +0.0063 |
| Mean Dice | 0.8955 | 0.9007 | +0.0052 |
| WT HD95 | 4.91 | 5.41 | +0.50 |
| TC HD95 | 3.26 | 3.73 | +0.47 |
| ET HD95 | 3.62 | 3.49 | -0.14 |

## 6. 当前可用于汇报的结论

1. 在统一的 sliding-window 全体积测试下，`TAF-TransBTS A3` 是当前整体 Dice 最佳模型，Mean Dice 为 `0.9007`，首次超过 `0.90`。
2. MSA 将 WT Dice 提高到 `0.9169`，说明多模态注意力值得保留，但需要避免牺牲 TC。
3. RAM 和 RAMS 表明区域导向融合能够修复部分困难病例，空间注意力也能改善边界稳定性，但整体收益尚不稳定。
4. RSF 揭示了一个重要问题：更强的区域监督不一定带来更好的泛化，过强的病灶响应会造成远端假阳性和 HD95 恶化。
5. TAF A3 说明受控跨模态融合主体有效：WT、TC、ET 三项 Dice 均超过 baseline，其中 TC 和 ET 的收益更明显。
6. correlation-guided TAF 的稳定化代码已经准备完成，但尚未得到正式 A4 测试证据。后续应单独验证 A4，不能将 A3 的收益归因于 correlation loss。
7. TAF 的 WT 和 TC HD95 仍略逊于 baseline，后续可结合病例级分析和 connected-component 后处理判断误差来源。

## 7. 最终 TAF 证据文件

本轮提交保留最终可复核证据：

- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap/checkpoints/best.ckpt`：epoch `150` 最佳权重，通过 Git LFS 跟踪。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap/training_log.csv`：从 epoch `31` 到 `200` 的续跑曲线。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap/train_summary.json`：最佳 epoch 与 full-val Mean Dice 汇总。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap/early_stop.json`：early stopping 状态。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap/metrics_test.json`：`127` 个测试病例的最终指标。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap_train.log`：完整训练终端日志。
- `outputs/taf_transbts_cluster_a3_finetune_resume_mmap_eval_test.log`：完整测试终端日志，记录 sliding-window `127/127` 完成。

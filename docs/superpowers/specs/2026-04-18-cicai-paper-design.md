# CICAI 2026 论文设计文档

**题目**: Uncertainty-Aware Multi-Modal Intent Inference for Human-Swarm Interaction
**会议**: CAAI International Conference on Artificial Intelligence (CICAI 2026)
**截稿**: 2026-05-25
**类型**: 算法论文（有实验）

---

## 1. 问题定义

多模态人机交互系统中，语音、手势、触屏三路输入信号本质上是模糊的（识别置信度不稳定、模态间可能冲突）。现有系统采用简单阈值过滤或单模态优先策略，在噪声条件下误触发率高、鲁棒性差。

**研究问题**：如何将三路不确定性模态信号可靠地映射为精确的无人机集群控制指令？

---

## 2. 核心贡献

1. 提出自适应模态可靠性估计器，动态调整各模态权重
2. 设计基于贝叶斯推断的意图推断模型，融合历史指令上下文先验
3. 引入三档决策门控机制，显式处理不确定区间，降低误触发率
4. 构建多模态无人机指令数据集（7种模态组合 × 4种噪声条件）

---

## 3. 方法设计

### 3.1 问题形式化

每时刻 $t$ 收到三路观测：
- $o_v^t$：语音识别结果 + 置信度 $c_v \in [0,1]$
- $o_g^t$：手势识别结果 + 置信度 $c_g \in [0,1]$
- $o_s^t$：触屏形状识别结果 + 置信度 $c_s \in [0,1]$

目标：推断 $I^* = \arg\max_{I \in \mathcal{C}} P(I \mid o_v, o_g, o_s, \text{history})$

指令集合 $\mathcal{C}$：{起飞, 降落, 悬停, 上升, 下降, 前进, 编队, 确认}（8类）

### 3.2 三层架构

**Layer 1 — Per-Modality Reliability Estimator**

动态评估每个模态的可靠性权重 $w_m \in [0,1]$：

$$w_m^t = \alpha \cdot \text{acc}_m^{\text{recent}} + (1-\alpha) \cdot f(\text{env}_m^t)$$

- $\text{acc}_m^{\text{recent}}$：最近 N 次该模态的历史准确率（滑动窗口）
- $f(\text{env}_m^t)$：环境特征函数（语音：背景噪声估计；手势：光照/遮挡估计；触屏：轨迹点数）
- $\alpha$：平衡历史与实时环境的超参数

**Layer 2 — Bayesian Intent Inference**

$$P(I \mid o_v, o_g, o_s) \propto P(o_v \mid I)^{w_v} \cdot P(o_g \mid I)^{w_g} \cdot P(o_s \mid I)^{w_s} \cdot P(I \mid \text{history})$$

- 似然 $P(o_m \mid I)$：由各检测器置信度参数化的类条件概率
- 先验 $P(I \mid \text{history})$：基于最近 $k$ 条指令的一阶马尔可夫转移概率（从训练数据统计）

**Layer 3 — Decision Gate**

$$\text{action} = \begin{cases} \text{执行} I^* & \text{if } P(I^* \mid \cdot) > \theta_{\text{high}} \\ \text{请求确认} & \text{if } \theta_{\text{low}} < P(I^* \mid \cdot) \leq \theta_{\text{high}} \\ \text{拒绝，提示重输} & \text{if } P(I^* \mid \cdot) \leq \theta_{\text{low}} \end{cases}$$

---

## 4. 实验设计

### 4.1 数据集

基于现有 multimodal_detector 系统采集：

| 维度 | 设置 |
|------|------|
| 指令类别 | 8类 |
| 模态组合 | 7种（单×3 + 双×3 + 三×1） |
| 噪声条件 | 正常 / 低光照 / 背景噪声 / 手势遮挡 |
| 每类样本 | ~50 samples |
| 总计 | ~1600 条 |

### 4.2 Baseline

| 方法 | 描述 |
|------|------|
| Single-Voice | 仅语音模态 |
| Single-Gesture | 仅手势模态 |
| Majority Vote | 三模态简单多数投票 |
| Fixed-Weight Fusion | 固定权重加权融合 |
| **Ours** | 自适应贝叶斯意图推断 |

### 4.3 评估指标

- **Command Accuracy**：正确指令识别率
- **False Trigger Rate**：误触发率
- **Rejection Rate**：正确拒绝模糊输入比例
- **Latency**：端到端响应延迟（ms）

### 4.4 消融实验

| 变体 | 去掉的模块 |
|------|-----------|
| w/o Adaptive Weight | 固定权重替代自适应权重 |
| w/o Context Prior | 去掉历史指令先验（均匀先验） |
| w/o Decision Gate | 直接 argmax，无门控 |
| **Full Model** | 完整方法 |

---

## 5. 论文结构

1. Introduction
2. Related Work（多模态融合、意图推断、UAV HRI）
3. System Overview（multimodal_detector 作为实验平台）
4. Method（三层架构）
5. Experiments
6. Conclusion

---

## 6. 实验平台

- 系统：multimodal_detector v2.14
- 路径：`/home/ubuntu/NGW/intern/multimodal_detector`
- 检测器：Whisper (voice) + MediaPipe (gesture) + OpenCV contour (touch)
- 无人机仿真：PyQtGraph OpenGL 3D + APF 避障 + L1 动力学模型

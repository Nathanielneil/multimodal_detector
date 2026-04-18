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

### 3.0 统一指令本体（Command Ontology）

三路模态的原始输出需先映射到统一指令空间 $\mathcal{C}$，再进行融合：

| 统一指令 $I$ | 语音关键词 | 手势 | 触屏形状 |
|-------------|-----------|------|---------|
| TAKEOFF | 起飞/升空 | 张开手掌 | — |
| LAND | 降落/着陆 | 握拳 | — |
| HOVER | 悬停/停住 | 食指指向 | — |
| ALT_UP | 上升/高一点 | 竖起大拇指 | — |
| ALT_DOWN | 下降/低一点 | 向下大拇指 | — |
| MOVE_FWD | 向前飞/forward | V形手势 | — |
| FORMATION | 编队（+类型） | ILY手势 | 三角/方形/圆形/五角星 |
| CONFIRM | 确认/好的 | OK手势 | — |

- 触屏形状（三角/方形/圆形/五角星）统一映射为 FORMATION，子类型作为附加参数
- 某模态无对应输出时，该模态置信度设为 0，不参与融合

### 3.1 问题形式化

每时刻 $t$ 收到三路观测：
- $o_v^t$：语音识别结果，置信度 $c_v \in [0,1]$（经校准，见 3.1.1）
- $o_g^t$：手势识别结果，置信度 $c_g \in [0,1]$（经校准）
- $o_s^t$：触屏形状识别结果，置信度 $c_s \in [0,1]$（经校准）

目标：推断 $I^* = \arg\max_{I \in \mathcal{C}} P(I \mid o_v, o_g, o_s, \text{history})$

#### 3.1.1 置信度校准

三路检测器原始输出不可直接比较，需统一校准为概率估计：

| 模态 | 原始输出 | 校准方法 |
|------|---------|---------|
| 语音 (Whisper) | `avg_logprob` ∈ (-∞, 0] | Platt scaling: sigmoid(a·logprob + b)，参数从验证集拟合 |
| 手势 (MediaPipe) | 规则匹配硬编码值 {0.85, 0.9} | 用验证集混淆矩阵估计类条件准确率，替换硬编码值 |
| 触屏 (OpenCV) | 固定 1.0 | 用形状识别的几何评分（圆度/顶点匹配度）替代固定值 |

校准后的置信度 $c_m \in [0,1]$ 作为似然 $P(o_m \mid I)$ 的参数化估计。

### 3.2 三层架构

**Layer 1 — Per-Modality Reliability Estimator**

动态评估每个模态的可靠性权重 $w_m \in [0,1]$：

$$w_m^t = \alpha \cdot \hat{r}_m^t + (1-\alpha) \cdot f(\text{env}_m^t)$$

- $\hat{r}_m^t$：在线可靠性估计，用跨模态一致性作为代理信号（当多模态一致时，各模态可靠性上升；冲突时下降）。无需外部 ground truth。
- $f(\text{env}_m^t)$：环境特征函数（语音：RMS 背景噪声估计；手势：帧亮度均值；触屏：轨迹点数归一化）
- $\alpha$：超参数，在验证集上搜索

**Layer 2 — Product-of-Experts Intent Inference**

采用 Product-of-Experts (PoE) 框架（Hinton, 2002），各模态作为独立专家。权重先归一化：

$$\tilde{w}_m = \frac{w_m}{\sum_{m'} w_{m'}}$$

推断公式：

$$P(I \mid o_v, o_g, o_s) \propto P(I)^\beta \cdot \prod_{m \in \{v,g,s\}} P(I \mid o_m)^{\tilde{w}_m}$$

- $\tilde{w}_m$ 归一化后满足 $\sum \tilde{w}_m = 1$，避免先验指数为负
- $\beta \in [0,1]$ 为先验强度超参数，在验证集上搜索
- 每个专家 $P(I \mid o_m)$ 由校准后的置信度参数化（softmax over 指令类别）
- 先验 $P(I)$ 使用历史指令的一阶马尔可夫转移概率（Laplace 平滑处理稀疏转移）
- 归一化权重保证 PoE 公式数学一致性，$\beta$ 控制先验影响强度

**Layer 3 — Decision Gate**

阈值 $\theta_{\text{high}}, \theta_{\text{low}}$ 在验证集 ROC 曲线上选取，以最大化 F1 为准：

$$\text{action} = \begin{cases} \text{执行} I^* & \text{if } P(I^* \mid \cdot) > \theta_{\text{high}} \\ \text{请求确认} & \text{if } \theta_{\text{low}} < P(I^* \mid \cdot) \leq \theta_{\text{high}} \\ \text{拒绝，提示重输} & \text{if } P(I^* \mid \cdot) \leq \theta_{\text{low}} \end{cases}$$

---

## 4. 实验设计

### 4.1 数据集

基于现有 multimodal_detector 系统采集，单人操作员录制：

| 维度 | 设置 |
|------|------|
| 指令类别 | 8类（见 3.0 指令本体） |
| 模态条件 | 单模态×3 + 多模态组合×2（语音+手势、手势+触屏） = 5种 |
| 噪声条件 | 正常 / 低光照 / 背景噪声 / 手势遮挡 |
| 每类×每条件样本 | ~30 samples |
| 总计 | 8类 × 5模态条件 × 4噪声条件 × 30 = **4800 条** |
| 模糊样本集 | 双模态冲突（如语音说 TAKEOFF，手势做 LAND）额外采集 200 条，人工标注"应执行/应拒绝" |
| 划分 | 60% 训练 / 20% 校准+验证（阈值调优与 Platt scaling 分别使用不同子集，避免数据泄漏）/ 20% 测试 |

**噪声条件复现方式**：
- 低光照：关闭主灯，仅保留环境光（约 50 lux）
- 背景噪声：扬声器播放标准化白噪声（60 dB SPL）
- 手势遮挡：黑色遮挡板覆盖手部约 40% 面积

**跨模态一致性定义**（用于 Layer 1 可靠性估计）：
- 若多模态同时激活且映射到同一指令类别 $I$，则各模态一致性分数 +1
- 若映射到不同类别，则冲突模态一致性分数 -1，一致模态不变
- 滑动窗口（最近 10 次）平均后归一化到 [0,1] 作为 $\hat{r}_m^t$

### 4.2 Baseline

| 方法 | 描述 |
|------|------|
| Single-Voice | 仅语音模态 |
| Single-Gesture | 仅手势模态 |
| Majority Vote | 三模态简单多数投票 |
| Fixed-Weight Fusion | 固定权重加权融合 |
| **Ours** | 自适应贝叶斯意图推断 |

### 4.3 评估指标

- **Command Accuracy**：正确指令识别率（测试集，排除拒绝样本）
- **False Trigger Rate**：低置信度输入被错误执行的比例
- **Rejection F1**：在人工标注的模糊样本集上，拒绝类别的 F1（同时考虑 Precision 和 Recall，避免"全拒绝"策略刷高 Precision）
- **Latency**：端到端响应延迟（ms），Whisper 使用 `tiny` 模型，实验中报告实际延迟数值

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

- 系统：multimodal_detector v2.14（仿真环境，无真实无人机）
- 路径：`/home/ubuntu/NGW/intern/multimodal_detector`
- 检测器：Whisper tiny (voice) + MediaPipe (gesture) + OpenCV contour (touch)
- 无人机仿真：PyQtGraph OpenGL 3D + APF 避障 + L1 动力学模型（纯软件仿真）
- 硬件：Ubuntu 20.04，CPU 实验（GPU 可选加速 Whisper）
- 实验范围：所有结果来自仿真平台，不涉及真实无人机飞行

---

## 7. 相关工作方向（待补充文献）

- 多模态融合：早期/晚期/混合融合综述
- 不确定性量化：Bayesian deep learning, conformal prediction in HRI
- UAV 人机交互：gesture-based UAV control, voice command for drones
- Product-of-Experts：Hinton (2002), 后续在多模态学习中的应用

---

## 8. 时间规划

| 周次 | 任务 |
|------|------|
| Week 1 (4/18–4/25) | 实现置信度校准 + PoE 融合模块 + 决策门控 |
| Week 2 (4/26–5/2) | 数据采集（2880 条 + 200 条模糊样本） |
| Week 3 (5/3–5/9) | 运行实验、消融实验、阈值调优 |
| Week 4 (5/10–5/18) | 撰写论文（含 Related Work） |
| Week 5 (5/19–5/25) | 润色、格式检查、提交 |

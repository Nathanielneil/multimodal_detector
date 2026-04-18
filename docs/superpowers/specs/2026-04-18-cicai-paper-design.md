# CICAI 2026 论文设计文档

**题目**: Uncertainty-Aware Multi-Modal Intent Inference for Human-Swarm Interaction
**会议**: CAAI International Conference on Artificial Intelligence (CICAI 2026)
**截稿**: 2026-05-25
**类型**: 算法论文（有实验）

---

## 1. 问题定义

多模态人机交互系统中，语音、手势、触屏三路输入信号本质上是模糊的（识别置信度不稳定、模态间可能冲突）。现有系统采用简单阈值过滤或单模态优先策略，在噪声条件下误触发率高、鲁棒性差。

**研究问题**：如何将三路不确定性模态信号，结合集群实时状态，可靠地映射为精确的无人机集群控制指令？

---

## 2. 核心贡献

1. 提出自适应模态可靠性估计器，融合跨模态一致性与语义感知时序稳定性，动态调整各模态权重
2. 设计带指令条件相关性修正的 PoE 意图推断模型，融合从飞行日志学习的集群状态先验与动态先验强度
3. 引入三档决策门控机制，显式处理不确定区间，并通过 TCT/Communication Efficiency 量化交互代价
4. 构建多用户多模态无人机指令数据集（实采 ~7680 条 + 增强至 14400 条，含 600 条冲突样本，6名受试者 LOOCV）
5. 在 AirSim 高保真仿真中验证方法，展示环境退化下的权重动态迁移与确认闭环行为

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

$$w_m^t = \alpha \cdot \hat{r}_m^t + (1-\alpha) \cdot f(\text{env}_m^t, \text{stab}_m^t)$$

- $\hat{r}_m^t$：跨模态一致性代理信号，采用**加权移动平均 (WMA)** 替代简单滑动窗口，当前帧权重最高，向过去指数衰减（衰减系数 $\lambda$ 在验证集上搜索）。论文中明确报告各检测器帧率（Whisper ~2Hz，MediaPipe ~30Hz）及 WMA 引入的理论延迟上界（ms），证明在 HSI 允许的实时范围内。
- $f(\text{env}_m^t, \text{stab}_m^t)$：联合环境与时序稳定性评分，环境特征通过**非线性 Sigmoid 映射**转换为可靠性分数（通过小规模校准实验拟合，使权重在进入"识别红区"时迅速切断该模态）：
  - 语音：$\sigma(-k_v \cdot (\text{RMS} - \tau_v))$，$k_v, \tau_v$ 从校准实验拟合
  - 手势：$\sigma(-k_g \cdot (\tau_g - \text{亮度均值}))$，低光照时迅速降权
  - 触屏：轨迹点数归一化（线性，触屏对环境不敏感）
  - **时序稳定性惩罚**（所有模态共用）：若某模态在最近 5 帧内输出类别发生跳变，惩罚强度由**语义相似度矩阵** $S_{IJ}$ 决定：语义相反的跳变（LAND→TAKEOFF）重罚，语义无关的跳变（MOVE_FWD→FORMATION）中罚，合法快速重复（ALT_UP→ALT_UP）不惩罚。$S_{IJ}$ 从指令本体的语义关系预定义。
- $\alpha$：超参数，在验证集上搜索

**Layer 2 — Correlation-Corrected PoE Intent Inference**

采用 Product-of-Experts (PoE) 框架（Hinton, 2002），并引入模态相关性修正解决独立性假设导致的 over-confidence 问题。权重先归一化：

$$\tilde{w}_m = \frac{w_m}{\sum_{m'} w_{m'}}$$

相关性修正：将静态矩阵升级为**指令条件相关性** $\rho_{mn}(I) \in [0,1]$，从训练数据中按指令类别分别估计。**稀疏性处理**：对样本量不足的低频指令（如 ALT_DOWN），以全局平均相关性 $\bar{\rho}_{mn}$ 作为层级先验初始值，并将语义相近指令（ALT_UP / ALT_DOWN）的相关性矩阵加权融合，缓解稀疏估计问题：

$$\rho_{mn}(I) = \frac{N_I \cdot \hat{\rho}_{mn}(I) + N_0 \cdot \bar{\rho}_{mn}}{N_I + N_0}$$

其中 $N_I$ 为指令 $I$ 的训练样本数，$N_0$ 为平滑超参数。

$$\tilde{w}_m^{\text{eff}}(I) = \tilde{w}_m \cdot \left(1 - \max_{n \neq m} \rho_{mn}(I) \cdot \tilde{w}_n\right)$$

推断公式（含集群状态先验）：

$$P(I \mid o_v, o_g, o_s, \mathbf{s}) \propto P(I \mid \mathbf{s})^{\beta(\mathbf{s})} \cdot \prod_{m \in \{v,g,s\}} P(I \mid o_m)^{\tilde{w}_m^{\text{eff}}(I)}$$

**归一化注意**：由于 $\tilde{w}_m^{\text{eff}}(I)$ 对不同 $I$ 取值不同，归一化（Softmax）前必须对所有 $I \in \mathcal{C}$ 完整计算各自的 $\rho_{mn}(I)$，再统一归一化，否则概率分布失去物理意义。

- $P(I \mid \mathbf{s})$：**学习得到的集群状态条件先验**，用轻量 MLP（2层，隐层 32 维）从 AirSim 飞行日志中学习 $P(I_t \mid I_{t-1}, \mathbf{s}_t)$。训练数据来源多样化：人工遥控轨迹 + 最优控制（A*）轨迹 + 随机扰动轨迹，避免 MLP 仅学习单一控制策略的近似。论文中强调 MLP 捕捉的是集群物理状态（编队连通度、电量非线性变化）与意图之间的复杂非线性关系，这是硬编码规则难以覆盖的。
- $\beta(\mathbf{s})$：**动态先验强度**，与集群风险等级挂钩：平稳飞行时 $\beta$ 小（尊重人意图），低电量/紧急避障时 $\beta$ 大（强制干预）。**Safety Break 机制（带迟滞）**：进入条件为某模态置信度 $c_m > 0.95$ 且推断指令与先验最高概率指令完全相反，且该冲突持续 $K \geq 3$ 帧；退出条件为 $c_m < 0.85$（迟滞比较器防止高频抖动/Chattering）。触发后强制将 $\beta$ 降至 $\beta_{\min}$，触发"紧急人工接管"，保障 Human-in-the-loop 安全性。
- 每个专家 $P(I \mid o_m)$ 由校准后的置信度参数化（softmax over 指令类别）

**Layer 3 — Decision Gate**

阈值 $\theta_{\text{high}}, \theta_{\text{low}}$ 通过 **False Trigger Rate vs. TCT 的 Pareto 曲线** 在验证集上选取，而非单纯最大化 F1。具体做法：在验证集上扫描 $(\theta_{\text{high}}, \theta_{\text{low}})$ 的网格，绘制 False Trigger Rate（安全性）与 TCT（交互效率）的权衡曲线，选取 Pareto 前沿上满足 False Trigger Rate $< \epsilon$ 约束的最小 TCT 点。$\epsilon$ 由任务安全需求决定（论文中设 $\epsilon = 0.05$）。

$$\text{action} = \begin{cases} \text{执行} I^* & \text{if } P(I^* \mid \cdot) > \theta_{\text{high}} \\ \text{请求确认} & \text{if } \theta_{\text{low}} < P(I^* \mid \cdot) \leq \theta_{\text{high}} \\ \text{拒绝，提示重输} & \text{if } P(I^* \mid \cdot) \leq \theta_{\text{low}} \end{cases}$$

**Layer 2.5 — Parameter Consensus（带参数指令的子参数融合）**

对于带子参数的指令（当前仅 FORMATION），在 Layer 2 确定 $I^* = \text{FORMATION}$ 后，对各模态提供的子参数（形状类型）进行加权投票：

$$\text{param}^* = \arg\max_{p} \sum_{m: o_m \text{ 含参数}} \tilde{w}_m^{\text{eff}}(I^*) \cdot \mathbf{1}[o_m.\text{param} = p]$$

- 若仅一个模态提供参数（如只有触屏划出三角形），直接采用该参数
- 若多模态参数冲突（如触屏三角形 vs 语音圆形），按 $\tilde{w}_m^{\text{eff}}$ 加权投票，票数相同时触发 Layer 3 请求确认
- 指令空间 $\mathcal{C}$ 正式定义为 $\{\text{Action} \times \text{Parameter}\}$，其中 Parameter 对非 FORMATION 指令为空

---

### 4.1 数据集

基于现有 multimodal_detector 系统采集，**6名受试者**（不同性别、手势幅度、语音语调）：

| 维度 | 设置 |
|------|------|
| 指令类别 | 8类（见 3.0 指令本体） |
| 模态条件 | 单模态×3 + 多模态组合×2（语音+手势、手势+触屏） = 5种 |
| 噪声条件 | 正常 / 低光照 / 背景噪声 / 手势遮挡 |
| 受试者 | 6名 |
| 每类×每条件×每人样本（实际采集） | ~8 samples |
| 实际采集总量 | 8类 × 5模态条件 × 4噪声条件 × 6人 × 8 ≈ **7680 条** |
| 数据增强后总量 | ~**14400 条**（语音加噪、手势图像旋转/遮挡、触屏轨迹加高斯噪声） |
| 模糊样本集 | 双模态冲突额外采集 **600 条**（每人 100 条，占比提升），人工标注"应执行/应拒绝" |
| 划分 | **LOOCV（留一用户法）**：每次以 1 名受试者为测试集，其余 5 名为训练+验证集（Platt scaling 校准使用验证集的独立子集，避免数据泄漏） |
| 虚拟受试者增强 | 基于 6 人数据生成 4 名虚拟受试者（语音 Pitch Shift ±20%、手势骨架噪声注入），用于测试极端边缘情况，不参与主实验 LOOCV |
| 用户分群分析 | 在结果部分分析"专家用户"与"新手用户"在 $w_m$ 分布上的差异，展示系统对不同用户的自适应能力 |

**噪声条件复现方式**：
- 低光照：关闭主灯，仅保留环境光（约 50 lux）
- 背景噪声：扬声器播放标准化白噪声（60 dB SPL）
- 手势遮挡：黑色遮挡板覆盖手部约 40% 面积

**跨模态一致性定义**（用于 Layer 1 可靠性估计，WMA 版本）：
- 若多模态同时激活且映射到同一指令类别 $I$，则各模态一致性分数 +1
- 若映射到不同类别，则冲突模态一致性分数 -1，一致模态不变
- 使用 WMA 加权平均（当前帧权重最高，指数衰减），归一化到 [0,1] 作为 $\hat{r}_m^t$

**数据采集工具**：开发自动化录制脚本，支持实时标注（操作员按键记录真实意图），减少人工标注工作量。

### 4.2 Baseline

| 方法 | 描述 |
|------|------|
| Single-Voice | 仅语音模态 |
| Single-Gesture | 仅手势模态 |
| Majority Vote | 三模态简单多数投票 |
| Fixed-Weight Fusion | 固定权重加权融合 |
| Cross-Attention Fusion | Transformer cross-attention 多模态融合（黑盒对比） |
| **Ours** | 相关性修正 PoE + 集群状态先验 + 自适应权重 |

### 4.3 评估指标

- **Command Accuracy**：正确指令识别率（测试集，排除拒绝样本）
- **False Trigger Rate**：低置信度输入被错误执行的比例
- **Rejection F1**：在人工标注的模糊样本集上，拒绝类别的 F1
- **Task Completion Time (TCT)**：从用户发出意图到集群完成响应的端到端时间，含"请求确认"交互的额外延迟
- **Communication Efficiency**：成功执行指令数 / 总交互轮次（衡量确认机制的代价）
- **Latency**：推断模块端到端延迟（ms），不含 AirSim 仿真执行时间

### 4.4 消融实验

| 变体 | 去掉的模块 |
|------|-----------|
| w/o Adaptive Weight | 固定权重替代自适应权重 |
| w/o Correlation Correction | 去掉相关性修正（退化为标准 PoE） |
| w/o Swarm Prior | 去掉集群状态先验（均匀先验） |
| w/o Temporal Stability | 去掉时序稳定性惩罚 |
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

- 系统：multimodal_detector v2.14（多模态输入端）
- 仿真后端：**AirSim**（高保真物理仿真，支持风力干扰、光影遮挡）
- 接口：multimodal_detector 推断结果通过 ROS Bridge 发布到 AirSim 集群控制节点
- 检测器：Whisper tiny (voice) + MediaPipe (gesture) + OpenCV contour (touch)
- 集群规模：6架无人机
- 通信延迟测试：人工注入 50ms / 100ms / 200ms 随机延迟
- **AirSim 环境-性能关联实验**：调整光照（正常→低光）和风力（0→5m/s），记录 Layer 1 权重 $w_g$（手势）的动态下降过程，展示决策权自动转移至语音模态的行为
- **AirSim 感知延迟注入**：除通信延迟外，注入感知延迟（模拟无人机高速运动导致图像传输模糊，反馈给检测器的置信度下降），测试系统在感知退化下的鲁棒性
- **确认闭环仿真**：在 AirSim 中完整模拟"系统请求确认 → 用户 OK 手势 → 系统执行"流程，以 Communication Efficiency（成功执行数/总交互轮次）作为用户负荷的代理指标
- **可视化输出**：置信度-权重动态演化图（三路信号随环境噪声波动的时序图），作为论文核心图表
- 硬件：Ubuntu 20.04，CPU 实验（GPU 可选加速 Whisper）

---

## 7. 相关工作方向（待补充文献）

- 多模态融合：早期/晚期/混合融合综述，Cross-Attention Transformer 融合
- 不确定性量化：Bayesian deep learning, conformal prediction in HRI
- UAV 人机交互：gesture-based UAV control, voice command for drones, human-swarm interaction
- Product-of-Experts：Hinton (2002), 后续在多模态学习中的应用
- 集群控制：swarm state estimation, formation control feedback

---

## 8. 时间规划

| 周次 | 任务 |
|------|------|
| Week 1 (4/18–4/25) | 实现置信度校准 + 条件相关性修正 PoE + 集群状态先验 MLP + 动态 β + 决策门控；开发数据采集自动化脚本 |
| Week 2 (4/26–5/2) | 数据采集（6名受试者，实采 ~7680 条 + 600 条冲突样本）；数据增强；AirSim 接口集成 |
| Week 3 (5/3–5/9) | 运行 LOOCV 实验、消融实验、阈值调优；AirSim 环境退化实验 + 确认闭环仿真；生成置信度-权重动态演化图 |
| Week 4 (5/10–5/18) | 撰写论文（含 Related Work） |
| Week 5 (5/19–5/25) | 润色、格式检查、提交 |

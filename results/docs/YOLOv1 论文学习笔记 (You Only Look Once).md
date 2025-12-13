# YOLOv1 论文学习笔记 (You Only Look Once)

## 1. 核心思想与范式转变

### 1.1 传统目标检测 vs YOLO
*   **传统方法 (两阶段/多阶段，如 R-CNN)**：
    1.  **Region Proposal**：先在图像上生成大量的候选框（“这里可能有物体”）。
    2.  **Classification**：对切出来的候选框进行分类（“这是什么物体”）。
    3.  **缺点**：流程割裂，速度慢，无法进行端到端的全局优化。

*   **YOLO (一阶段/Regression)**：
    *   将目标检测重构为一个单一的**回归问题 (Regression Problem)**。
    *   **One-stage**：输入图像，直接一次性输出所有包围框的坐标 $(x, y, w, h)$ 和类别概率。
    *   **全局视野**：训练和测试时，YOLO 都能看到完整的图像。
        *   它可以学习类别的**上下文信息 (Contextual Information)**，利用 $P(\text{物体} | \text{环境})$ 进行推断。
        *   相比 Fast R-CNN，YOLO 的**背景误检率 (Background Errors)** 更低。

### 1.2 优缺点概览
*   **优点**：
    *   **速度快**：45 FPS (Base) / 155 FPS (Fast)，满足实时性。
    *   **泛化能力强**：能够理解物体与环境的逻辑关系。
*   **缺点**：
    *   **细节丢失**：由于经过多次下采样且没有滑动窗口，对细节关注不足（更偏向宏观）。
    *   **小物体与聚集物体难检测**：受限于“一个格子只能预测一组类别”，如果一个格子里有多个物体中心（如群鸟），模型无法处理。

---

## 2. 预测流程设计 (The Grid System)

### 2.1 网格划分与张量维度
1.  **Grid 划分**：讲输入图片分割为 $S \times S$ 个网格（Paper中 $S=7$）。
    *   **核心逻辑**：若某物体的**中心点 (Center)** 落在该网格中，该网格就负责检测该物体。

2.  **预测输出**：每个网格需要预测 $B$ 个边界框信息 和 $C$ 个类别概率。
    *   **Bounding Box ($B=2$)**：每个框包含 5 个参数：
        *   $x, y$：框中心相对于**当前格子**边缘的偏移 ($0 \sim 1$)。
        *   $w, h$：框的宽高相对于**整张图片**的比例 ($0 \sim 1$)。
        *   $Confidence$：置信度。
    *   **Class Probability ($C=20$)**：条件概率 $P(Class_i | Object)$。注意这是**针对格子**的，不是针对每个框的。

3.  **Tensor 维度计算**：
    $$
    S \times S \times (B \times 5 + C) \Rightarrow 7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30
    $$

### 2.2 置信度与评分公式
*   **预测时的框置信度**：
    $$
    Confidence = P(Object) \times IOU_{pred}^{truth}
    $$
    *   含义：该网格有物体的概率 $\times$ 预测框与真实框的重合度。

*   **类别的条件概率**：
    $$
    P(Class_i | Object)
    $$
    *   含义：**假如**这个格子里有物体，它是猫/狗/车的概率。

*   **测试时的最终得分 (Score)**：
    将上述两部分相乘，得到每个框针对每个类别的特定得分：
    $$
    \text{Score}_{ij} = P(Class_i | Object) \times (P(Object) \times IOU) = P(Class_i) \times IOU
    $$

---

## 3. 网络设计与训练策略

### 3.1 架构特点
YOLO 的流程既是黑箱（特征提取不可见）又是白箱（输出结构被严格定义）。
*   **卷积层 (Convolution)**：
    *   作用：提取特征。
    *   **$1 \times 1$ 卷积**：用于降维（压缩通道数），减少计算量，增加非线性。
    *   **$3 \times 3$ 卷积**：用于特征提取。
*   **池化层 (Pooling)**：
    *   作用：降低空间尺寸（Downsampling），扩大**感受野 (Receptive Field)**。
    *   逻辑：如果不缩小图片，卷积核只能看到局部；池化后，卷积核一扫就能覆盖物体整体结构。
*   **输出端**：全连接层强制输出 $7 \times 7 \times 30$ 的张量，前10位为回归坐标，后20位为分类概率。

### 3.2 训练策略
1.  **预训练 (Pre-training)**：先在 ImageNet 上训练分类网络（使用 $224 \times 224$ 输入）。
2.  **微调 (Fine-tuning)**：转为检测任务。
    *   增加 4 层卷积 + 2 层全连接。
    *   **分辨率大跃进**：输入从 $224 \times 224$ 调整为 **$448 \times 448$**。
    *   *原因*："Detection often requires fine-grained visual information"（检测需要更精细的纹理和轮廓信息）。

---

## 4. 损失函数的设计 (Loss Function)

YOLO 的损失函数采用了均方误差 (Sum-Squared Error, SSE) 的变体，针对目标检测的特殊性进行了四点关键修正。

### 4.1 核心问题与对策

#### 问题 1：正负样本极度不平衡 (Background >>> Foreground)
*   **现象**：一张图中绝大多数格子是背景（没物体）。
*   **后果**：背景格子的梯度会淹没前景格子，导致模型倾向于预测“全无”。
*   **对策**：引入权重系数 $\lambda$。
    *   **$\lambda_{coord} = 5$**：大幅增强对**有物体**坐标预测的惩罚。
    *   **$\lambda_{noobj} = 0.5$**：大幅降低对**无物体**置信度预测的惩罚。

#### 问题 2：大框小框的不公平性
*   **现象**：同样的像素偏差（如偏离 5px），对大物体影响小，对小物体影响致命。直接预测 $w, h$ 会导致 loss 对尺度不敏感。
*   **对策**：预测 $\sqrt{w}, \sqrt{h}$。
    *   **原理**：利用 $\sqrt{x}$ 函数的特性（在 $x$ 较小时斜率大，较大时斜率小），放大这一偏差对小物体的影响。

#### 问题 3：谁来负责 (Responsible Predictor)
*   **机制**：一个格子预测 2 个框，但在训练时，只有一个框能去拟合 Ground Truth。
*   **选择标准**：与真实框 **IoU 最大** 的那个预测框。
*   **效果**：促进预测器的**专业化 (Specialization)**。
    *   *演化过程*：通过正反馈循环，某些节点会专门对“细长物体”敏感，另一些对“扁宽物体”敏感。这是无监督的自动分工。

### 4.2 损失函数公式详解

总 Loss = 
$$
\begin{aligned}
& \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \quad & \leftarrow \text{中心点坐标误差} \\
+ & \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \quad & \leftarrow \text{宽高误差(根号)} \\
+ & \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 \quad & \leftarrow \text{置信度误差(有物体)} \\
+ & \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 \quad & \leftarrow \text{置信度误差(无物体)} \\
+ & \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2 \quad & \leftarrow \text{类别概率误差}
\end{aligned}
$$

*   $\mathbb{1}_{ij}^{obj}$：表示第 $i$ 个网格的第 $j$ 个 bbox 负责预测该物体（Responsible）。
*   注意：分类误差只在网格中有物体 ($\mathbb{1}_{i}^{obj}$) 时才计算。

---

## 5. 数学基础补遗：卷积 vs 池化

定义输入特征图 $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$，输出 $\mathbf{Y}$。

### 5.1 卷积 (Convolution)
*   **本质**：空间加权求和 + 通道全融合。
*   **公式**：
    $$ \mathbf{Y}_{i,j,k} = \sigma \left( b_k + \sum_{c=1}^{C_{in}} \sum_{u, v} \mathbf{X}_{pos, c} \cdot \mathbf{W}_{u, v, c}^{(k)} \right) $$
*   **特点**：
    *   $\sum_{c=1}^{C_{in}}$ 意味着卷积会把输入的所有通道信息融合。
    *   输出通道数 $C_{out}$ 取决于卷积核的数量，与输入通道无关（深度重映射）。
    *   主要用于提取特征。

### 5.2 最大池化 (Max Pooling)
*   **本质**：非线性降采样。
*   **公式**：
    $$ \mathbf{Y}_{i,j,c} = \max_{window} \left( \mathbf{X}_{pos, c} \right) $$
*   **特点**：
    *   **Channel-wise**：通道之间独立运算，互不干扰。
    *   输出通道数 $C_{out} = C_{in}$（深度保持）。
    *   主要用于降低维度、扩大感受野、保持平移不变性。
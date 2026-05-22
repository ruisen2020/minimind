# 位置编码：从 Sinusoidal PE 到 RoPE

> **本文目标**：深入理解 Transformer 中位置编码的演进历程，从绝对位置编码到相对位置编码的本质变革。

---

# 第一部分：Sinusoidal 位置编码

## 一、为什么需要位置编码？

Transformer 的自注意力机制是**置换不变的**（permutation invariant）：

```
输入序列: ["我", "爱", "你"]  →  Attention计算
输入序列: ["你", "爱", "我"]  →  Attention计算结果相同！
```

这意味着模型无法区分 "我爱你" 和 "你爱我"——这显然不对！

**位置编码的作用**：为每个 token 注入位置信息，让模型知道"这个词在第几个位置"。

---

## 二、Sinusoidal PE 的核心公式

Transformer 原论文提出了一种基于正弦/余弦函数的位置编码：

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**关键符号解释**：

| 符号 | 含义 | 取值范围 |
|------|------|----------|
| `pos` | token 在序列中的位置 | 0, 1, 2, ..., seq_len-1 |
| `i` | embedding 的维度索引 | 0, 1, ..., d_model/2 - 1 |
| `2i` 和 `2i+1` | 偶数维和奇数维 | 成对使用 sin/cos |

**直观理解**：
- **偶数维度**：用正弦函数编码
- **奇数维度**：用余弦函数编码
- 不同维度使用不同频率（i 越大，频率越低）

---

## 三、手算示例：理解 PE 的计算过程

**假设**：$d_{model}=8$，计算第 2 个位置（pos=2）的编码向量。

### Step 1：计算每个维度的缩放因子

| i | 对应维度 | $\text{div\_term}_i = 10000^{2i/d_{model}}$ |
|---|----------|---------------------------------------------|
| 0 | 0, 1 | $10000^{0} = 1$ |
| 1 | 2, 3 | $10000^{0.25} \approx 10$ |
| 2 | 4, 5 | $10000^{0.5} = 100$ |
| 3 | 6, 7 | $10000^{0.75} \approx 1000$ |

### Step 2：逐维计算

$$
\begin{aligned}
\text{PE}(2, 0) &= \sin(2/1) \approx 0.9093 \\
\text{PE}(2, 1) &= \cos(2/1) \approx -0.4161 \\
\text{PE}(2, 2) &= \sin(2/10) \approx 0.1987 \\
\text{PE}(2, 3) &= \cos(2/10) \approx 0.9801 \\
\text{PE}(2, 4) &= \sin(2/100) \approx 0.0200 \\
\text{PE}(2, 5) &= \cos(2/100) \approx 0.9998 \\
\text{PE}(2, 6) &= \sin(2/1000) \approx 0.0020 \\
\text{PE}(2, 7) &= \cos(2/1000) \approx 0.9999
\end{aligned}
$$

### Step 3：得到位置编码向量

```
pos=2 的位置编码: [0.9093, -0.4161, 0.1987, 0.9801, 0.0200, 0.9998, 0.0020, 0.9999]
                   ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^
                   低频(变化快)  ←──────────────────────→  高频(变化慢)
```

**观察**：低维度（i=0）数值变化剧烈，高维度（i=3）数值几乎不变——这就是**远程衰减**特性。

---

## 四、代码实现

```python
import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
    """
    Sinusoidal 位置编码实现
    
    参数:
        seq_len: 序列长度
        d_model: 模型维度
    返回:
        pe: (seq_len, d_model) 的位置编码矩阵
    """
    # pos 索引：[0, 1, 2, ..., seq_len-1]，形状 (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]
    
    # 频率缩放因子：[10000^0, 10000^{2/d}, 10000^{4/d}, ...]
    div_term = np.power(10000, (2 * np.arange(d_model // 2) / d_model))
    
    # 初始化编码矩阵
    pe = np.zeros((seq_len, d_model))
    
    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = np.sin(position / div_term)  # 第0, 2, 4, ... 维
    pe[:, 1::2] = np.cos(position / div_term)  # 第1, 3, 5, ... 维
    
    return pe

# 测试
pe = sinusoidal_position_encoding(seq_len=120, d_model=8)
print(pe.shape)  # (120, 8)
print(f"pos=2 的编码: {pe[2]}")
```

---

## 五、远程衰减特性

### 什么是远程衰减？

> **远程衰减**：随着维度索引 i 增大，相邻位置之间的编码差异变小。

**数学直觉**：
- 公式中 $10000^{2i/d_{model}}$ 在分母
- i 越大 → 分母越大 → 角度变化越慢 → 编码差异越小

### 量化验证

假设 $d_{model}=512$，$i=256$（高维），比较 pos=1000 和 pos=1001：

$$
\Delta = \sin(0.1001) - \sin(0.1000) \approx 0.0000003
$$

**结论**：位置差 1，高维编码几乎不变！这意味着模型在高层维度难以区分相邻位置。

### 可视化验证

```python
import matplotlib.pyplot as plt

seq_len, d_model = 512, 128
pe = sinusoidal_position_encoding(seq_len, d_model)

plt.figure(figsize=(10, 6))
for i in [0, 32, 64]:  # 不同维度索引
    plt.plot(range(seq_len), pe[:, i], label=f'i={i}')

plt.xlabel("Position (pos)")
plt.ylabel("Encoding Value")
plt.title("远程衰减：高维变化慢，低维变化快")
plt.legend()
plt.show()
```

**解读图表**：
- `i=0`（蓝色）：波形密集，相邻位置差异大
- `i=64`（绿色）：波形平缓，相邻位置差异小

---

## 六、Sinusoidal PE 的缺陷

### 缺陷一：只能提供绝对位置

```
Attention 计算: Q · K = (x + PE_m) · (y + PE_n)
```

位置编码通过**加法**融入，模型需要自己学习如何从绝对位置提取相对位置信息。

### 缺陷二：长序列外推能力差

| 场景 | 问题 |
|------|------|
| 训练 | 只见过 pos ∈ [0, 1024] 的位置模式 |
| 推理 | 输入长度 4096，出现未见过的位置模式 |
| 结果 | 性能下降（频率过高，数值剧烈变化）|

---

# 第二部分：旋转位置编码（RoPE）

## 一、RoPE 的核心思想

> **核心洞察**：与其把位置信息"加"到 embedding 上，不如通过"旋转"把位置信息融入 Q 和 K 的计算中。

**RoPE 的目标**：找到一种变换，使得：
$$
f(q, m) \cdot f(k, n) = g(q, k, m-n)
$$

即：**注意力分数只依赖相对位置差 (m-n)**，而非各自的绝对位置。

---

## 二、从二维旋转说起

### 复习：二维向量旋转

给定向量 $\mathbf{x} = [x, y]^T$，逆时针旋转角度 $\theta$：

$$
R(\theta) \cdot \mathbf{x} = 
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
$$

**关键性质**：旋转矩阵正交，模长不变，只改变方向。

### 为什么旋转能编码位置？

假设 Q 在位置 m，K 在位置 n：

1. 对 Q 旋转角度 $m\theta$：$\tilde{q} = R(m\theta) \cdot q$
2. 对 K 旋转角度 $n\theta$：$\tilde{k} = R(n\theta) \cdot k$
3. 计算注意力分数：

$$
\tilde{q}^T \tilde{k} = q^T R(-m\theta) R(n\theta) k = q^T R((n-m)\theta) k
$$

**神奇之处**：注意力分数只依赖 $(n-m)\theta$，即相对位置！

---

## 三、二维示例：直观理解 RoPE

**假设**：
- $q = [1, 2]^T$，位置 m=1
- $k = [3, 4]^T$，位置 n=2
- 角频率 $\omega = 1$

### 计算过程

```python
import numpy as np

q = np.array([1, 2])
k = np.array([3, 4])
omega = 1.0
m, n = 1, 2

# 计算旋转角度
theta_q = omega * m  # = 1
theta_k = omega * n  # = 2

# 旋转矩阵
def rotate(x, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ x

# 应用旋转
q_rot = rotate(q, theta_q)
k_rot = rotate(k, theta_k)

# 注意力分数
score = q_rot @ k_rot
print(f"RoPE 注意力分数: {score:.4f}")

# 验证：等价于用相对位置旋转
relative_theta = theta_k - theta_q  # = 1
k_rot_relative = rotate(k, relative_theta)
score_check = q @ k_rot_relative
print(f"相对位置验证: {score_check:.4f}")  # 应该相同
```

**结果**：两种方式计算的分数相同，证明了 RoPE 的相对位置特性。

---

## 四、扩展到高维向量

### 分组旋转策略

对于 $d_{model}$ 维向量，**每两个维度一组**，独立旋转：

```
向量: [x0, x1, x2, x3, x4, x5, ...]
       └───┘  └───┘  └───┘
       第0组   第1组   第2组
       θ0      θ1      θ2
```

每组使用不同的角频率：
$$
\theta_i = \frac{1}{10000^{2i/d_{model}}}
$$

### 高维旋转公式

$$
\text{RoPE}(x, m) = 
\begin{bmatrix}
x_0 \cos(m\theta_0) - x_1 \sin(m\theta_0) \\
x_0 \sin(m\theta_0) + x_1 \cos(m\theta_0) \\
x_2 \cos(m\theta_1) - x_3 \sin(m\theta_1) \\
x_2 \sin(m\theta_1) + x_3 \cos(m\theta_1) \\
\vdots
\end{bmatrix}
$$

### 高效实现形式

为避免稀疏矩阵运算，改写为：

$$
\text{RoPE}(x, m) = x \odot \cos(\Theta) + \text{rotate\_half}(x) \odot \sin(\Theta)
$$

其中 `rotate_half` 定义为：

```python
def rotate_half(x):
    """将后半部分取负并交换"""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

# 示例
# 输入: [1, 2, 3, 4]
# 输出: [-3, -4, 1, 2]
```

---

## 五、手算示例：4维向量

**已知**：
- $q = [1, 0, 0, 1]$
- $d_{model} = 4$
- $pos = 2$
- $\omega = 10000$

### Step 1：计算角频率

$$
\theta_0 = \frac{1}{10000^{0/4}} = 1.0, \quad \theta_1 = \frac{1}{10000^{2/4}} = 0.01
$$

### Step 2：计算旋转角度

$$
\alpha_0 = pos \cdot \theta_0 = 2.0, \quad \alpha_1 = pos \cdot \theta_1 = 0.02
$$

### Step 3：逐组旋转

**第1组** $[x_0, x_1] = [1, 0]$，角度 2.0：
$$
\begin{bmatrix} 1 \cos(2) - 0 \sin(2) \\ 1 \sin(2) + 0 \cos(2) \end{bmatrix}
= \begin{bmatrix} -0.4161 \\ 0.9093 \end{bmatrix}
$$

**第2组** $[x_2, x_3] = [0, 1]$，角度 0.02：
$$
\begin{bmatrix} 0 \cos(0.02) - 1 \sin(0.02) \\ 0 \sin(0.02) + 1 \cos(0.02) \end{bmatrix}
= \begin{bmatrix} -0.02 \\ 0.9998 \end{bmatrix}
$$

### Step 4：拼接结果

$$
\text{RoPE}(q) = [-0.4161, 0.9093, -0.02, 0.9998]
$$

### PyTorch 验证

```python
import torch

d_model = 4
pos = 2
omega = 10000.0

# 计算频率
freqs = 1.0 / (omega ** (torch.arange(0, d_model, 2).float() / d_model))
angles = pos * freqs  # [2.0, 0.02]

# 构造 cos/sin 向量（复制一份）
cos = torch.cat([torch.cos(angles), torch.cos(angles)])
sin = torch.cat([torch.sin(angles), torch.sin(angles)])

# rotate_half 函数
def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

# 应用 RoPE
q = torch.tensor([1.0, 0.0, 0.0, 1.0])
q_embed = q * cos + rotate_half(q) * sin
print(q_embed)
# tensor([-0.4161, -0.0200,  0.9093,  0.9998])
```

---

## 六、MiniMind/LLaMA 的 RoPE 实现

```python
import torch

def precompute_freqs_cis(d_model: int, max_seq_len: int = 32768, base: float = 10000.0):
    """
    预计算所有位置的 cos/sin 值
    
    参数:
        d_model: 模型维度
        max_seq_len: 最大序列长度
        base: 频率基数（默认10000）
    返回:
        freqs_cos: (max_seq_len, d_model)
        freqs_sin: (max_seq_len, d_model)
    """
    # 计算角频率: θ_i = 1 / base^{2i/d}
    freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    
    # 位置索引: [0, 1, 2, ..., max_seq_len-1]
    t = torch.arange(max_seq_len)
    
    # 外积: 每个位置 × 每个频率 → 得到每个位置的角度
    # freqs: (d_model/2,) → angles: (max_seq_len, d_model/2)
    angles = torch.outer(t, freqs)
    
    # 复制扩展: (max_seq_len, d_model/2) → (max_seq_len, d_model)
    freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    对 Q 和 K 应用旋转位置编码
    
    参数:
        q: (batch, seq_len, n_heads, head_dim)
        k: (batch, seq_len, n_heads, head_dim)
        cos: (max_seq_len, head_dim)
        sin: (max_seq_len, head_dim)
    返回:
        q_embed, k_embed: 旋转后的 Q 和 K
    """
    def rotate_half(x):
        """将后半部分取负并移到前面"""
        d = x.shape[-1] // 2
        return torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    
    # 扩展维度以匹配 q/k 的形状
    cos = cos.unsqueeze(unsqueeze_dim)  # (1, max_seq_len, 1, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # 应用旋转: x * cos + rotate(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# ============ 使用示例 ============
if __name__ == "__main__":
    d_model = 64
    batch_size, seq_len, n_heads = 2, 10, 4
    head_dim = d_model // n_heads
    
    # 模拟 Q 和 K
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # 预计算 cos/sin
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len=1024)
    
    # 取当前序列长度对应的 cos/sin
    cos = freqs_cos[:seq_len]
    sin = freqs_sin[:seq_len]
    
    # 应用 RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"Q 原始形状: {q.shape}")
    print(f"Q 旋转后形状: {q_rot.shape}")  # 形状不变
```

### 关键点解析

| 函数 | 作用 |
|------|------|
| `precompute_freqs_cis` | 提前计算好所有位置的 cos/sin，推理时直接查表 |
| `torch.outer(t, freqs)` | 外积，得到每个位置在每个维度上的旋转角度 |
| `rotate_half` | 将向量后半部分取负移到前面，实现旋转操作的向量形式 |
| `unsqueeze_dim=1` | 扩展维度以适配多头注意力的形状 |

---

## 七、RoPE vs Sinusoidal 对比

| 特性 | Sinusoidal PE | RoPE |
|------|---------------|------|
| 引入方式 | 加法：x + PE | 乘法：旋转 Q/K |
| 位置感知 | 绝对位置 | **相对位置**（核心优势）|
| 外推能力 | 较差 | 较好（周期性旋转）|
| 参数量 | 0（固定编码）| 0（固定编码）|
| 计算方式 | 预计算 + 加 | 预计算 + 乘 |

---

## 八、总结

```
┌─────────────────────────────────────────────────────────────┐
│                    位置编码演进路线                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sinusoidal PE              RoPE                           │
│   ─────────────              ────                           │
│   • 加法融合                  • 旋转融合                     │
│   • 绝对位置                  • 相对位置                     │
│   • 外推能力差                • 外推能力强                   │
│                                                             │
│   核心思想转变:                                              │
│   "把位置加到 embedding" → "用位置旋转 Q/K"                  │
│                                                             │
│   数学本质:                                                  │
│   R(mθ) · R(nθ)^T = R((n-m)θ)                              │
│   旋转的逆 = 反向旋转 → 相对位置自动出现                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**RoPE 的本质贡献**：
> 以绝对位置编码的形式，实现了相对位置感知能力。

---

## 参考资料

- [Transformer 原论文](https://arxiv.org/abs/1706.03762)
- [RoPE 原论文](https://arxiv.org/abs/2104.09864)
- [苏剑林博客：RoPE 详解](https://kexue.fm/archives/8265)
- [HuggingFace RoPE 实现讨论](https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509)

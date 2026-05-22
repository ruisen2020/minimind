# 一、稠密模型中的FFN
稠密模型（Dense Model）是指模型结构中所有的参数和计算路径在每次前向计算中都会被激活。

相比于原始Transformer中使用的FFN，这里的FFN引入了Gated结构，可以动态的控制信息流通（比如抑制/强调特定特征），增加了表达能力。
原始FFN的表达能力有限，非线性太简单了。

我们可以引入Gated控制信息流，它有两个分支，一个分支做内容，一个分支做门控。这个门控就可以有选择地放大或者抑制信息。
有的信息放大，有的信息官小。
这样针对不同的token，会激活不同的通道，即有线性也有非线性，比传统FFN更加灵活。

举个例子：
标准FFN：
FFN(x) = ReLU(xW1)W2
非线性来自ReLU
ReLU(x) = max(0, x)
非线性是硬开关，要么全关，要么全开

而Gated FFN是 FFN(x) = (xW1) * sigmoid(xW2)
sigmoid的输出是(0 ~ 1 之间的连续值)

所以Gated FFN 的控制方式更灵活，能够动态调节。


![alt text](image-7.png)

我们来讲一讲具体的实现细节：

首先有两个分支：
gate_proj：做门控
up_proj：做内容

gate_proj:生成门控信号，控制强弱
```python
self.gate_proj = nn.Linear(hidden, intermediate)
```
act_fn：激活函数，比如 sigmoid、relu等等
self.act_fn(self.gate_proj(x))

up_proj:内容分支
生成内容特征

down_proj：把高维标识压回 hidden_size

self.act_fn(self.gate_proj(x)) * self.up_proj(x)
这里会把得到的门控矩阵 是直接和self.up_proj(x)相乘，也就是说每个元素和内容矩阵中的元素直接向乘。
而且传统FFN是直接进行线性变换，让两个矩阵相乘，特征之间互相混合。
但是门控FFN中是对每个特征单独缩放，可以实现精细的特征选择与控制。



```python
# 激活函数映射字典
ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu
}

class FeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config["intermediate_size"] is None:
            intermediate_size = int(config["hidden_size"] * 8 / 3)
            # 为了更好地利用 GPU 的并行计算能力（特别是 TensorCore、SIMD 等），中间维度通常会做 64 对齐
            # 向上取整到最近的 64 的倍数
            config["intermediate_size"] = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.dropout = nn.Dropout(config["dropout"])
        self.act_fn = ACT2FN[config["hidden_act"]]

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )

# 示例配置
config = {
    "hidden_size": 512,
    "intermediate_size": None,  # 自动计算
    "dropout": 0.1,
    "hidden_act": "silu"  # 也可以试试 "gelu", "relu"
}

# 创建模型和输入
ffn = FeedForward(config)
x = torch.randn(2, 16, 512)  # (batch_size, seq_len, hidden_size)

# 前向传播
out = ffn(x)
print("Output shape:", out.shape) # Output shape: torch.Size([2, 16, 512])
```



# 二、稀疏模型MOE

稠密模型适合中小规模模型结构，部署和训练更简单稳定，是大多数基础模型的核心形式。

但如果想让模型变得更大、更强，而又不想付出巨额推理开销。可以使用稀疏模型（MOE），这是一种更具扩展性的结果，其核心思想是：模型参数很多，但每次只激活其中一部分，通过门控机制选择专家进行预测，最后组合专家输出。

简单来说，传统的FFN只全量计算，如果想增加模型的能力，就要增加FFN的参数量，但是增加参数量必然会增加训练和推理开销。而且增加参数，那么每个token还是公用同一个FFN进行计算，表达能力也有限。
所以有没有更好的方式，即增加能力，又不增加开销，也能提高表达能力。

答案就是MOE，混合专家网络。它可以增加FFN的参数量，但是每次使用的时候，只使用一部分参数，这样推理开销也没有增加。两全其美。针对不同的token，使用不同的参数，这样表达能力也大大增加。
这就好比，有很多条路，每次不需要把路全走一遍，只走一条即可。

那么有这么多条路，选哪条路呢，所以需要有一个路由模块，来每个token找到最合适的路。
这个路由模块是怎么知道token最适合哪个路呢？

那么就需要说这些专家都是这么构造出来的，路由器是如何路由的。

在训练开始时，所有的专家和路由都是随机的参数。
那么一个token遇到多个专家，就碰巧某个专家的路由分数最高，那么就去更新这个专家，并且更新这个路由。
如果发现通过这个专家的损失函数降低了，那么路由就会增大这种token的概率，让更多类似的token进入到这条路上。
由于训练数据集很大，所以每个专家都有可能被选中，即使选错也没有关系，损失函数增大了，那么被选中的概率也会下降。这就是自然选择。

所以在训练结束之后，路由就能够很好的知道每个token适合什么专家，每个专家也能更好的服务token。相辅相成。

但是还是有可能出问题，假设出现某个专家特别强，所有token 都路由到了同一个专家，那么这样就会退化成为传统的FFN，路由就失效了。表达能力也退化了。

怎么办呢，这就需要做好负载均衡。
load balance loss的作用是：防止所有token都挤到少数几个专家，导致其他专家空闲，模型能力浪费。

MOE 本质会路由到 top k 个专家，但由于参数都是随机初始化和自由选择的，会导致好的专家越来越好，差的专家越来越差。

最后会导致：
1.专家浪费
2.表达能力下降
3.训练不稳定

负载均衡就想办法让每个专家被选中的概率有一样。
它会增加一个损失函数，损失函数为每个专家被选中的概率乘上实际被选中的次数，这样如果某个专家概率很大，实际被选中次数也很多，那么就会导致损失函数增加。

也有的会是 标准差/平均值 来作为损失函数。







为什么 MoE 需要 load balance loss
Top-1 vs Top-2 routing 区别
Switch Transformer 是怎么简化 MoE 的
MoE + KV Cache 推理怎么做
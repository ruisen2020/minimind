# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# RMSNorm 归一化
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    #  0, 1, 2, ..., d/2-1 
    # torch.arange(0, dim, 2) = torch.tensor([0, 2, 4, ..., (d-1)/2]) 获取从（0，dim）之间的偶数
    # [: (dim // 2)]： 取前d/2个元素
    # 这产生从低频到高频的频率序列，低频对应长距离依赖，高频对应短距离依赖。
    # freps 计算基础频率：dim维度的偶数索引位置
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # YaRN 扩展机制（用于长度外推） 先不看这个
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    #  t = [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 计算旋转角度 t * freqs
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 应用旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

# 复制kv, 用于多头自注意力
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        #  支持多query-head共享同一个key/value-head
        # num_key_value_heads: 每个token的key和value的头数，用于多头自注意力 GQA：group query attention
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 保证num_attention_heads是num_key_value_heads的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # n_local_heads: 总的 attention head 数
        self.n_local_heads = args.num_attention_heads
        # n_local_kv_heads: 每个 token 的 key 和 value 的头数
        self.n_local_kv_heads = self.num_key_value_heads
        # n_rep: 每个 k/v head 被多少个 query head 共享
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # head_dim: 每个 head 的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        # QKV 线性层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # 输出线性层
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 是否使用flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        # x: (bsz, seq_len, hidden_size)
        bsz, seq_len, _ = x.shape
        # 注意力计算 QKV
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 转换为多头自注意力的输入格式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        # 将历史kv与当前kv拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # KV head 重复扩展 -> 让所有 Q head 对应到正确的 KV head
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 计算注意力分数
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用flash attention
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 使用标准的attention

            # 计算注意力分数 
            # scores: (bsz, n_local_heads, seq_len, seq_len)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 实现causal mask, 将上三角设置为负无穷，这样就不会选择未来token
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 注意力掩码
            if attention_mask is not None:
                # [bsz,seq_len]-->[bsz,1,1,seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # [bsz,1,1,seq_len]-->[bsz,1,1,seq_len]
                # 需要掩的位置是0，现在变成了-1e9，非常小
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                # 需要掩码的位置变成负无穷后，后面计算softmax的时候，负无穷的位置会变成0，不会选择
                scores = scores + extended_attention_mask

            # 计算注意力分数
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # dropout
            scores = self.attn_dropout(scores)
            # 加权求和 [bsz, n_local_heads, seq_len, seq_len] @ [bsz, n_local_heads, seq_len, head_dim] = [bsz, n_local_heads, seq_len, head_dim]
            output = scores @ xv
        # 恢复为原始的输入格式 [bsz, seq_len, n_local_heads, head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 输出线性层
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 门控线性层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 下采样线性层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 上采样线性层
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 计算门控线性层和上采样线性层的输出，然后相乘，再经过激活函数，最后经过下采样线性层和dropout
        # 这里的“门控机制”具体的逻辑是这样的：

        # 首先，通过 gate = act_fn(gate_proj(x)) 得到门控信号（范围通常为[0, 1]或正数）；

        # 其次，通过 value = up_proj(x) 得到要被控制的信息通道；

        # 接着，使用 gated = gate * value 应用门控，控制信息强度（本质上就是加权）；

        # 最后，使用output = down_proj(gated) 降维回去，门控完成。
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 每个token被路由到的专家数量
        self.top_k = config.num_experts_per_tok
        # 总的专家数量
        self.n_routed_experts = config.n_routed_experts

        # 专家选择函数，也就是每个专家选中的概率
        self.scoring_func = config.scoring_func
        # 辅助损失权重
        self.alpha = config.aux_loss_alpha
        # 是否使用序列级辅助损失
        self.seq_aux = config.seq_aux
        # 是否对topk概率进行归一化
        self.norm_topk_prob = config.norm_topk_prob
        # 专家选择函数的输入维度
        self.gating_dim = config.hidden_size
        # 专家选择函数的权重 [n_routed_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [bsz, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.shape
        # [bsz * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)

        # [bsz * seq_len, n_routed_experts] 计算每个token被路由到每个专家的概率
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # [bsz * seq_len, top_k] 选择每个token被路由到的专家的索引和权重
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            # 对topk概率进行归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 如果是训练模式，且辅助损失权重大于0，就需要计算辅助损失
        if self.training and self.alpha > 0.0:
            # [bsz * seq_len, n_routed_experts]
            scores_for_aux = scores
            # topk： 每个token被路由到的专家数量
            aux_topk = self.top_k
            # [bsz, seq_len * top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # seq_aux： 是否使用序列级辅助损失
            # 按 序列 维度分别计算每个样本内部的 expert 使用分布
            if self.seq_aux:
                # [bsz, seq_len, n_routed_experts]
                # 每个token被路由到每个专家的概率
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # [bsz, n_routed_experts]  初始化每个 batch 的 expert 统计图 [bsz, n_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                #  ce： 每个token被路由到每个专家的平均概率
                #  按 batch 维度分别计算每个样本内部的 expert 使用分布
                #  [bsz, seq_len * top_k]，每个样本所有 token 的 top_k expert 索引
                #  [bsz, seq_len * top_k]，topk_idx_for_aux_loss索引位置计数为1
                # 标准化：理论上每个 expert 的理想负载是平均的 => 除以 (seq_len * top_k / n_experts)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                
                # - scores_for_seq_aux.mean(dim=1): [bsz, n_experts]，每个样本对每个 expert 的平均打分
                # - ce: 每个 expert 的归一化使用频率，越接近1说明越平均
                # - ce * score：频率高且分数高则惩罚高
                # - 最终 loss 对 batch 取平均，乘 alpha
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 从整个batch维度统计专家的使用频率

                # one-hot 编码每个 token 被选中的 expert
                # [bsz * seq_len * top_k, n_routed_experts]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                #  [n_routed_experts]，每个 expert 被使用的频率（除以总 token 数量）
                ce = mask_ce.float().mean(0)
                # [n_experts]，所有 token 对各 expert 的平均打分
                Pi = scores_for_aux.mean(0)
                # 频率 × expert 数 = 负载比，理想负载为1，偏离表示不均
                fi = ce * self.n_routed_experts
                #  aux_loss = sum(Pi[i] * fi[i])，高打分 + 高负载的 expert 会被惩罚
                # - Pi: 平均打分，值越大说明 router 趋向于使用该 expert
                # - fi: 实际负载，值越大说明该 expert 被频繁使用
                # - Pi * fi：打分高且频率高的 expert 会导致更大的 loss（目标是让负载更均匀）
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 初始化专家列表
        # 每个专家都是一个FeedForward层
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 初始化门控网络
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        # x: [bsz, seq_len, hidden_size]
        identity = x
        # [bsz, seq_len, hidden_size]
        orig_shape = x.shape
        # [bsz, seq_len]
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        # topk_idx: [bsz * seq_len, top_k] 每个token被路由到的专家的索引
        # topk_weight: [bsz * seq_len, top_k] 每个token被路由到的专家的权重
        # aux_loss: 辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # [bsz * seq_len, hidden_size] 所有token
        x = x.view(-1, x.shape[-1])
        # [bsz * seq_len * top_k] 所有token被路由到的专家的索引
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 如果在训练阶段，每个token被复制topk次，然后送入专家进行处理，最后再按照权重进行加权平均
            # [bsz * seq_len * top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                # 对于每个专家，处理所有被路由到该专家的token
                expert_out = expert(x[flat_topk_idx == i])
                # 如果专家输出为空，需要添加一个梯度，否则会导致梯度消失
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            
            # y.view(*topk_weight.shape, -1): [bsz * seq_len * top_k, hidden_size]
            # topk_weight.unsqueeze(-1): [bsz * seq_len * top_k, 1] 每个专家对应的权重
            # 在top_k 维度上进行加权平均
            # y [bsz * seq_len, hidden_size]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 恢复y： [bsz, seq_len, hidden_size]
            y = y.view(*orig_shape)
        else:
            # 专家推理
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        #  加上共享专家的输出（可选）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                # identity 是原始的输入 每个共享专家都作用在原始输入上并加到输出中
                y = y + expert(identity)

        # 保存门控产生的辅助损失
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 推理节点的MOE前向传播，按照专家编号将token分组，分别送入专家进行处理后合并
        # x: [bsz * seq_len, hidden_size] 所有token
        # flat_expert_indices: [bsz * seq_len * top_k] 每个token被路由到的专家的索引
        # flat_expert_weights: [bsz * seq_len * top_k, 1] 每个专家对应的权重
        # 初始化输出缓存，用于存储每个专家处理的token的输出
        expert_cache = torch.zeros_like(x)
        # idxs: [bsz * seq_len * top_k] 按照专家索引排序后的token索引
        # 比如 flat_expert_indices = [2, 0, 1, 2, 1, 0]
        # token1 ： 专家2 专家0
        # token2: 专家1 专家2
        # token3: 专家1 专家0

        # 那么idxs = [1, 5, 2, 4, 3, 0]
        # 0: 1, 5  专家0处理的token1和token5
        # 1: 2, 4
        # 2: 0, 3
        # 这样是把token分发给不同的专家处理
        idxs = flat_expert_indices.argsort()
        # tokens_per_expert: [num_experts] 每个专家处理的token的数量,前缀和
        # tokens_per_expert ： [2,4,6]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # token_idxs: [bsz * seq_len * top_k] 每个token在所有token中的索引
        # idxs = [1, 5, 2, 4, 3, 0], top_k = 2,那么每个元素/2就是它在所有token中的索引
        # 那么token_idxs = [0, 2, 1, 2, 1, 0]
        # 专家0处理的token1和token5，token1在所有token中的索引为0，token5在所有token中的索引为2
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果start_idx == end_idx，说明没有token被路由到这个专家，直接跳过
            if start_idx == end_idx:
                continue
            # 当前是要处理哪个专家
            expert = self.experts[i]
            # 这个专家处理的token的索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 这个专家处理的token
            expert_tokens = x[exp_token_idx]
            # 这个专家处理的token的输出
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 这个专家处理的token的输出，乘以权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到缓存输出中，这样就可以把所有专家处理的token的输出累加到一起
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 多头注意力的头数
        self.num_attention_heads = config.num_attention_heads
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 多头注意力的头的维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 多头自注意力
        self.self_attn = Attention(config)

        # 当前层的id
        self.layer_id = layer_id
        # Attention 前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN 后的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈网络模块，可配置是否使用专家混合MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # hidden_states 输入的隐藏状态，是上一层的输出 [bsz, seq_len, hidden_size]
        # position_embeddings 位置编码，是RoPE编码 [seq_len, hidden_size]
        # past_key_value 上一层的key和value，用于缓存，加速推理 [bsz, num_heads, seq_len, head_dim]
        # use_cache 是否使用缓存，加速推理
        # attention_mask 注意力掩码，用于屏蔽padding位置

        # ==== self_attention ===

        # 1. 残差连接：将输入的隐藏状态复制一份，用于后续的加法操作。
        residual = hidden_states
        # 2. 自注意力层：对输入的隐藏状态进行自注意力计算，得到新的隐藏状态和当前层的key和value。
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), # 先进行输入前的归一化
            position_embeddings, # 位置编码
            past_key_value, use_cache, attention_mask # 缓存、注意力掩码
        )
        # 3. 残差连接：将自注意力层的输出与原始输入相加，得到新的隐藏状态。
        hidden_states += residual
        # self.post_attention_layernorm(hidden_states) ：FFN前的归一化
        # self.mlp(self.post_attention_layernorm(hidden_states))：FFN层，对归一化后的隐藏状态进行前馈处理，得到新的隐藏状态。
        # 再次进行残差连接 
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        # hidden_states：当前层的输出，用于下一层的输入。
        # present_key_value：当前层的key和value，用于缓存，加速推理
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 词表大小和层数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层 [词表大小，隐藏层大小] 词嵌入层，将输入的token转换为词向量
        # 这个是随机初始化的
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #  Dropout层，用于随机失活一些神经元，防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        #  Transformer层，包含多个MiniMindBlock，每个MiniMindBlock包含自注意力和前馈网络。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        #  最终输出前的归一化层，用于对隐藏状态进行归一化，防止梯度消失或爆炸
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 计算频率：预先计算所有位置的正弦和余弦值，避免每次前向传播重复计算。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # 将计算好的频率值注册为buffer，避免每次前向传播重复计算。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, # [bsz, seq_len]
                attention_mask: Optional[torch.Tensor] = None, # [bsz, seq_len]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # [bsz, num_heads, seq_len, head_dim]
                use_cache: bool = False, # 是否使用缓存，加速推理
                **kwargs):
        # input_ids 输入的token id [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 如果没有传入past_key_values，则初始化为None，长度为层数
        past_key_values = past_key_values or [None] * len(self.layers)
        # 获取历史缓存长度（增量生成时使用）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 隐藏状态：将输入的token id通过词嵌入层转换为词向量，然后通过Dropout层进行随机失活，得到隐藏状态。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 位置编码：从注册的buffer中获取位置编码，用于自注意力层。
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        #  Transformer层：对隐藏状态进行多层自注意力和前馈网络计算，得到新的隐藏状态和当前层的key和value。
        #  presents：当前层的key和value，用于缓存，加速推理
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # hidden_states：当前层的输出，用于下一层的输入。
            # present：当前层的key和value，用于缓存，加速推理
            hidden_states, present = layer(
                hidden_states, # 隐藏状态 [bsz, seq_len, hidden_size]
                position_embeddings,  # 位置编码 [seq_len, hidden_size]
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            # 将当前层的key和value添加到presents列表中，用于缓存，加速推理
            presents.append(present)
        # 最后一层的输出，进行归一化，得到最终的输出
        # [bsz, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)
        # 如果有MOE层，则计算MOE层的aux_loss，否则返回0
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# 定义MiniMindForCausalLM类，用于因果语言模型
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # MiniMindModel：Transformer模型，包含多个MiniMindBlock，每个MiniMindBlock包含自注意力和前馈网络。
        # 模型主干：MiniMindModel，输出hidden_states，presents，aux_loss
        self.model = MiniMindModel(self.config)
        # 输出层：将 hidden_size 映射为 vocab_size（即每个 token 的 logits）
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 将词嵌入层的权重和输出层的权重共享，加速训练
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # 这个参数 logits_to_keep 在 Causal LM（因果语言模型）里是一个性能优化 + 推理优化用的“截断控制参数”，核心作用是：
        # 🧠 控制“只计算最后 N 个 token 的 logits”，避免不必要的计算和显存浪费
        # 🚀 在推理时，可以设置 logits_to_keep=1，只计算最后一个 token 的 logits，加速推理
        # 📈 在训练时，设置 logits_to_keep=0，计算所有 token 的 logits，用于计算损失


        # 模型主干：MiniMindModel，输出hidden_states，presents，aux_loss
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 输出层：将 hidden_size 映射为 vocab_size（即每个 token 的 logits）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 从 h 中保留最后 logits_to_keep 个位置，送入 lm_head 做分类
        # 训练时，slice_indices 是 0，logits 相当于 self.lm_head(h[:, 0:, :])，即整个 h
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算损失：如果传入了labels，则计算交叉熵损失，否则返回None
        loss = None
        if labels is not None:
            # logits 去掉最后一个token，labels 去掉第一个token，计算交叉熵损失
            # shift 对齐成 预测下一个token
            # 例如：输入是 "I am a student"，labels是 "am a student"，logits是 "am a student"，则计算交叉熵损失
            # 所以必须要错位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # ignore_index=-100 表示忽略-100这个标签，不计算损失
            # 每个 token 都是一个 vocab 分类任务
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        # 将损失、logits、presents、hidden_states、aux_loss打包成CausalLMOutputWithPast对象，返回
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output

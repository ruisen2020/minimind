"""
pyq.csv 朋友圈广告二分类评测脚本
============================================
数据格式：id\ttext\tlabel（tab 分隔，无表头）
    - label=1：广告/带货推销内容
    - label=0：普通朋友圈分享

评测思路：
    1. 对每条文本构造分类 prompt，让模型输出 "0" 或 "1"
    2. 解析模型输出的第一个 0/1 数字作为预测
    3. 无法解析时记为 -1（视为预测错误）
    4. 统计 accuracy / precision / recall / f1 / 混淆矩阵

用法：
    # 全量评测，使用 full_sft 权重
    python eval_pyq.py --weight full_sft

    # 先采样 100 条快速跑通流程
    python eval_pyq.py --weight full_sft --sample_n 100

    # 使用 rlhf 权重 + 限制文本长度
    python eval_pyq.py --weight rlhf --max_text_len 300

    # 使用外部 transformers 模型
    python eval_pyq.py --load_from ./MiniMind2 --weight full_sft

    # 使用 Qwen2.5-1.5B-Instruct 评测（零训练方案，推荐）
    python eval_pyq.py --load_from Qwen/Qwen2.5-1.5B-Instruct --weight qwen
    python eval_pyq.py --load_from Qwen/Qwen2.5-1.5B-Instruct --weight qwen --sample_n 200
"""
import os
import re
import csv
import time
import argparse
import warnings
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# 模型初始化（与 eval_llm.py 保持一致）
# ---------------------------------------------------------------------------
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    # load_from 恰好等于 'model' 目录时走 MiniMind 原生 torch 权重加载；
    # 其他路径（如 HuggingFace hub id 'Qwen/Qwen2.5-1.5B-Instruct' 或本地目录）走 transformers 标准加载
    if args.load_from == 'model':
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
        model = model.to(args.device)
    else:
        # 外部模型（如 Qwen2.5-1.5B-Instruct）：使用 fp16/bf16 + device_map 避免 1.5B 参数爆显存
        dtype = torch.float16 if 'cuda' in args.device else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device)
        # Qwen 系列默认有 pad_token，但某些 tokenizer 没有，兜底一下
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    get_model_params(model, model.config)
    return model.eval(), tokenizer


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_pyq_csv(path: str):
    """
    读取 pyq.csv，返回 [(id, text, label), ...]
    文件格式：id\ttext\tlabel（tab 分隔，无表头）
    为了稳健性，用 csv.reader 并指定 tab 分隔符；若某行列数异常则跳过并计数。
    """
    samples = []
    bad_lines = 0
    # 用 utf-8 打开，quoting=QUOTE_MINIMAL 可以处理个别字段内含引号的情况
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) < 3:
                bad_lines += 1
                continue
            # 取前 3 列（若文本内意外含 tab，后面部分一并合到 text 里）
            _id = row[0].strip()
            # 如果列数 >3，认为 text 中含 tab，label 永远是最后一列
            _label_str = row[-1].strip()
            _text = '\t'.join(row[1:-1]).strip()
            if _label_str not in ('0', '1'):
                bad_lines += 1
                continue
            samples.append((_id, _text, int(_label_str)))
    print(f'📂 加载 {path} 完成：有效样本 {len(samples)} 条，跳过 {bad_lines} 条异常行')
    # 打印 label 分布
    cnt = Counter(lbl for _, _, lbl in samples)
    print(f'📊 label 分布：0（非广告）={cnt[0]}，1（广告）={cnt[1]}')
    return samples


# ---------------------------------------------------------------------------
# Prompt 构造 & 预测解析
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCT = (
    "你是一个严格的文本分类器。请判断给定的朋友圈内容是否为【广告/带货/商品推销】。"
    "只能回答一个数字：1 表示是广告/带货/推销，0 表示普通分享（非广告）。"
    "不要解释、不要输出任何其他字符。"
)


def build_prompt(text: str, max_text_len: int) -> str:
    """构造用户输入的文本分类 prompt；过长文本截断，避免占满上下文。"""
    if max_text_len > 0 and len(text) > max_text_len:
        text = text[:max_text_len] + '...'
    return (
        f"请判断下面这条朋友圈是否为广告/带货/推销内容，只回答 0 或 1：\n"
        f"<<<\n{text}\n>>>\n"
        f"答案（0 或 1）："
    )


def parse_prediction(response: str) -> int:
    """
    从模型回复中解析 0/1。
    策略：
        1. 找到回复中出现的第一个 0 或 1 数字字符
        2. 若都没有，返回 -1（标记为无效预测）
    """
    if response is None:
        return -1
    # 正则匹配第一个独立数字 0 或 1
    m = re.search(r'[01]', response)
    if m is None:
        return -1
    return int(m.group(0))


# ---------------------------------------------------------------------------
# 模型推理（单条）
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_one(model, tokenizer, args, text: str) -> (int, str):
    """对一条文本推理，返回 (pred, raw_response)"""
    setup_seed(2026)  # 分类任务固定种子，保证可复现

    user_prompt = build_prompt(text, args.max_text_len)

    # 外部模型（Qwen 等非 MiniMind）或非 pretrain 权重都走 chat_template；
    # 仅 MiniMind 的 pretrain 权重没训练过对话格式，直接拼 bos+文本
    is_external = (args.load_from != 'model')
    if is_external or args.weight != 'pretrain':
        conversation = [
            {"role": "system", "content": SYSTEM_INSTRUCT},
            {"role": "user", "content": user_prompt},
        ]
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = False  # 分类任务无需思考链，关掉节省 token
        input_text = tokenizer.apply_chat_template(**templates)
    else:
        input_text = tokenizer.bos_token + SYSTEM_INSTRUCT + '\n' + user_prompt

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=args.max_input_len).to(args.device)

    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,            # 分类任务贪心解码，确定性输出
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
    )
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return parse_prediction(response), response.strip()


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    手写 accuracy / precision / recall / f1（以 label=1 为正类），避免引入 sklearn 依赖。
    无效预测（-1）视为预测错误，统一按非正类（0）处理到混淆矩阵里，但也单独统计无效数量。
    """
    tp = fp = fn = tn = invalid = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == -1:
            invalid += 1
            # 无效预测按错误处理：若真实是 1 -> FN；若真实是 0 -> 也记 FP（视为乱答）
            if yt == 1:
                fn += 1
            else:
                fp += 1
            continue
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'total': total,
        'invalid': invalid,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def print_metrics(m):
    print('\n' + '=' * 60)
    print('📈 评测结果（正类 = 1 = 广告）')
    print('=' * 60)
    print(f"  样本总数         : {m['total']}")
    print(f"  无效预测数       : {m['invalid']}")
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  Precision (pos=1): {m['precision']:.4f}")
    print(f"  Recall    (pos=1): {m['recall']:.4f}")
    print(f"  F1        (pos=1): {m['f1']:.4f}")
    print('-' * 60)
    print('  混淆矩阵：')
    print(f"              预测0       预测1")
    print(f"  真实0    {m['tn']:>8}   {m['fp']:>8}")
    print(f"  真实1    {m['fn']:>8}   {m['tp']:>8}")
    print('=' * 60 + '\n')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型在 pyq.csv 上的广告二分类评测")
    # ----- 模型加载（与 eval_llm.py 保持一致） -----
    parser.add_argument('--load_from', default='model', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo, qwen）。"
                             "使用外部模型（如 Qwen）时该字段仅用于结果文件命名标识")
    parser.add_argument('--lora_weight', default='None', type=str,
                        help="LoRA权重名称（None表示不使用）")
    parser.add_argument('--hidden_size', default=768, type=int,
                        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用RoPE位置编码外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # ----- 评测相关 -----
    parser.add_argument('--csv_path', default='./pyq.csv', type=str, help="pyq.csv 文件路径")
    parser.add_argument('--out_csv', default='./pyq_eval_result.csv', type=str,
                        help="逐条预测结果的输出 CSV 路径")
    parser.add_argument('--sample_n', default=0, type=int,
                        help="评测样本数：0 表示全量，>0 随机采样 N 条（方便调试）")
    parser.add_argument('--seed', default=42, type=int, help="采样的随机种子")
    parser.add_argument('--max_text_len', default=500, type=int,
                        help="朋友圈文本字符级截断长度，避免超长样本拖慢评测（0 表示不截断）")
    parser.add_argument('--max_input_len', default=1024, type=int,
                        help="tokenizer 输入最大 token 长度")
    parser.add_argument('--max_new_tokens', default=8, type=int,
                        help="分类任务只需生成极少量 token，默认 8 足够")
    parser.add_argument('--log_every', default=50, type=int,
                        help="每多少条样本打印一次进度")
    args = parser.parse_args()

    # ---------- 1. 加载数据 ----------
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f'数据文件不存在：{args.csv_path}')
    samples = load_pyq_csv(args.csv_path)

    # 采样
    if args.sample_n and args.sample_n < len(samples):
        import random
        random.seed(args.seed)
        samples = random.sample(samples, args.sample_n)
        print(f'🔀 随机采样 {args.sample_n} 条进行评测（seed={args.seed}）')
        cnt = Counter(lbl for _, _, lbl in samples)
        print(f'📊 采样后 label 分布：0={cnt[0]}，1={cnt[1]}')

    # ---------- 2. 加载模型 ----------
    model, tokenizer = init_model(args)

    # ---------- 3. 逐条推理 ----------
    y_true, y_pred = [], []
    detail_rows = []  # 每条样本的明细，用于写出 CSV

    t0 = time.time()
    for idx, (sid, text, label) in enumerate(samples, 1):
        pred, raw = predict_one(model, tokenizer, args, text)
        y_true.append(label)
        y_pred.append(pred)
        detail_rows.append({
            'id': sid,
            'text': text.replace('\t', ' ').replace('\n', ' ')[:200],  # 明细只截前 200 字符，避免 CSV 过大
            'label': label,
            'pred': pred,
            'correct': int(pred == label),
            'raw_response': raw.replace('\t', ' ').replace('\n', ' ')[:100],
        })

        if idx % args.log_every == 0 or idx == len(samples):
            elapsed = time.time() - t0
            speed = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(samples) - idx) / speed if speed > 0 else 0.0
            # 动态准确率
            running_acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
            print(f'  [{idx:>5}/{len(samples)}] '
                  f'acc={running_acc:.4f} | '
                  f'speed={speed:.2f} it/s | '
                  f'elapsed={elapsed:.1f}s | ETA={eta:.1f}s')

    # ---------- 4. 计算指标 ----------
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # ---------- 5. 保存明细 ----------
    fieldnames = ['id', 'text', 'label', 'pred', 'correct', 'raw_response']
    with open(args.out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f'📝 逐条明细已保存到：{args.out_csv}')

    # ---------- 6. 保存汇总指标 ----------
    summary_path = args.out_csv.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'权重: {args.weight} (hidden_size={args.hidden_size}, moe={args.use_moe})\n')
        f.write(f'lora: {args.lora_weight}\n')
        f.write(f'数据: {args.csv_path}  样本数: {metrics["total"]}\n')
        f.write(f'max_text_len: {args.max_text_len}\n')
        f.write('-' * 40 + '\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
    print(f'📋 汇总指标已保存到：{summary_path}')


if __name__ == "__main__":
    main()

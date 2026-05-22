"""
广告分类评估脚本
===================
根据一个 CSV 文件（列：id, text, label，label∈{0,1}），使用 MiniMind 模型做二分类推理。
通过构造 Prompt 让模型输出 "0" 或 "1"，再和真实 label 做对比，统计准确率等指标。

用法示例：
    python eval_classify.py --csv_path data/ads.csv --weight full_sft --hidden_size 512

输出：
    1) 控制台打印 Accuracy / Precision / Recall / F1 / 混淆矩阵
    2) 生成 predictions.csv：id, text, label, pred, correct
"""
import os
import re
import time
import argparse
import warnings
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


# ============================================================
# 1. 模型初始化（完全沿用 eval_llm.py 的逻辑）
# ============================================================
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


# ============================================================
# 2. 构造分类 Prompt
# ============================================================
def build_prompt(text: str) -> str:
    """
    构造一个让模型做二分类的 Prompt。
    尽量简洁，要求模型只输出 0 或 1，方便后续解析。
    """
    return (
        "你是一个广告文本分类助手。请判断下面的文本是否为广告。\n"
        "如果是广告请回答 1，如果不是广告请回答 0。\n"
        "只需要输出一个数字（0 或 1），不要输出其他任何内容。\n\n"
        f"文本：{text}\n\n"
        "答案："
    )


# ============================================================
# 3. 从模型输出中解析 0 / 1
# ============================================================
def parse_label(response: str) -> int:
    """
    从模型回复中提取 0 或 1。
    - 优先匹配出现的第一个 0 或 1
    - 如果完全匹配不到，返回 -1（视为预测失败，统计时单独处理）
    """
    # 清理空白
    response = response.strip()

    # 直接匹配第一个 0 或 1 字符
    match = re.search(r'[01]', response)
    if match:
        return int(match.group())
    return -1


# ============================================================
# 4. 指标计算（Accuracy / Precision / Recall / F1 + 混淆矩阵）
# ============================================================
def compute_metrics(y_true, y_pred):
    """
    计算二分类指标。pred=-1 的样本视为"预测失败"，不计入 TP/TN/FP/FN，但计入 accuracy 分母。
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    failed = sum(1 for p in y_pred if p == -1)

    total = len(y_true)
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "failed": failed,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 50)
    print("📊 评估结果")
    print("=" * 50)
    print(f"总样本数     : {metrics['total']}")
    print(f"预测成功数   : {metrics['total'] - metrics['failed']}")
    print(f"预测失败数   : {metrics['failed']}  (模型未输出 0/1)")
    print(f"预测正确数   : {metrics['correct']}")
    print("-" * 50)
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"F1 Score    : {metrics['f1']:.4f}")
    print("-" * 50)
    print("混淆矩阵 (行：真实标签，列：预测标签)")
    print(f"             预测=0     预测=1")
    print(f"真实=0 (非广告)  {metrics['tn']:<8}  {metrics['fp']:<8}")
    print(f"真实=1 (广告)    {metrics['fn']:<8}  {metrics['tp']:<8}")
    print("=" * 50 + "\n")


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MiniMind 广告分类评估")
    # ---- 模型相关（和 eval_llm.py 保持一致）----
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA 权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用 RoPE 外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")

    # ---- 分类评估专属参数 ----
    parser.add_argument('--csv_path', required=True, type=str, help="输入 CSV 文件路径（列：id, text, label）")
    parser.add_argument('--output_path', default='predictions.csv', type=str, help="预测结果输出 CSV")
    parser.add_argument('--max_new_tokens', default=8, type=int, help="最大生成长度（分类任务只需几个 token）")
    parser.add_argument('--temperature', default=0.1, type=float, help="分类任务建议用低温度，让输出更确定")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus 采样阈值")
    parser.add_argument('--do_sample', default=0, type=int, choices=[0, 1], help="是否采样（分类建议 0，走 greedy）")
    parser.add_argument('--max_samples', default=-1, type=int, help="最多评估多少条（-1 表示全部）")
    parser.add_argument('--seed', default=2026, type=int, help="随机数种子")
    args = parser.parse_args()

    # 固定随机种子，结果可复现
    setup_seed(args.seed)

    # ---- 1) 读取 CSV ----
    print(f"📂 读取数据：{args.csv_path}")
    df = pd.read_csv(args.csv_path)
    required_cols = {'id', 'text', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列 {required_cols}，实际只有 {set(df.columns)}")
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    print(f"✅ 总样本数：{len(df)}")
    print(f"   类别分布：{df['label'].value_counts().to_dict()}")

    # ---- 2) 加载模型 ----
    model, tokenizer = init_model(args)
    print(f"✅ 模型加载完成，device = {args.device}\n")

    # ---- 3) 逐条推理 ----
    y_true, y_pred, raw_responses = [], [], []
    st = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
        text, gold = str(row['text']), int(row['label'])

        # 构造 prompt
        prompt = build_prompt(text)

        # 预训练模型用 bos + prompt，其他模型走 chat template
        if args.weight == 'pretrain':
            input_text = tokenizer.bos_token + prompt
        else:
            conversation = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(args.device)

        # 生成（分类任务用 greedy 更稳定）
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.0,
            )

        # 切掉输入部分，解码模型新生成的内容
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )

        pred = parse_label(response)
        y_true.append(gold)
        y_pred.append(pred)
        raw_responses.append(response.strip())

    elapsed = time.time() - st
    print(f"\n⏱  总耗时：{elapsed:.2f}s，平均每条：{elapsed / len(df) * 1000:.1f}ms")

    # ---- 4) 指标统计 ----
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # ---- 5) 保存每条预测结果 ----
    df_out = df.copy()
    df_out['pred'] = y_pred
    df_out['raw_response'] = raw_responses
    df_out['correct'] = [int(t == p) for t, p in zip(y_true, y_pred)]
    df_out.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f"💾 每条预测结果已保存到：{args.output_path}")


if __name__ == "__main__":
    main()

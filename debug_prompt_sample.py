"""
调试用：打印 eval_pyq.py 送进模型的完整 prompt 长什么样
不加载模型权重，只走 tokenizer 的 apply_chat_template 逻辑，秒级输出。

用法：
    python debug_prompt_sample.py
    python debug_prompt_sample.py --weight pretrain  # 看 pretrain 分支
"""
import argparse
from transformers import AutoTokenizer

# 与 eval_pyq.py 完全一致的 prompt 组装函数 ----------------------------------
SYSTEM_INSTRUCT = (
    "你是一个严格的文本分类器。请判断给定的朋友圈内容是否为【广告/带货/商品推销】。"
    "只能回答一个数字：1 表示是广告/带货/推销，0 表示普通分享（非广告）。"
    "不要解释、不要输出任何其他字符。"
)


def build_prompt(text: str, max_text_len: int = 500) -> str:
    if max_text_len > 0 and len(text) > max_text_len:
        text = text[:max_text_len] + '...'
    return (
        f"请判断下面这条朋友圈是否为广告/带货/推销内容，只回答 0 或 1：\n"
        f"<<<\n{text}\n>>>\n"
        f"答案（0 或 1）："
    )


# 3 条典型测试文本 ---------------------------------------------------------
DEMO_TEXTS = [
    ("10001", "Dior新款包包到货❗️秋冬经典款 💰1550/4300 有需要私我", 1),
    ("10004", "人到中年，才发现身体健康才是最大的财富，愿大家都平安健康[合十]", 0),
    ("10010", "朋友推荐的那本《当下的力量》真的不错，读完整个人都安静了下来，推荐给大家", 0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--weight', default='full_sft', type=str,
                        help='pretrain 走 else 分支，其他走 chat_template')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    for sid, text, label in DEMO_TEXTS:
        user_prompt = build_prompt(text, max_text_len=500)

        if args.weight != 'pretrain':
            conversation = [
                {"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": user_prompt},
            ]
            input_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = tokenizer.bos_token + SYSTEM_INSTRUCT + '\n' + user_prompt

        token_ids = tokenizer(input_text, return_tensors='pt')['input_ids'][0]

        print("=" * 80)
        print(f"📌 id={sid}  label={label}  weight={args.weight}")
        print("-" * 80)
        print("【最终送入模型的文本（repr 显示特殊字符）】")
        print(repr(input_text))
        print("-" * 80)
        print("【可读版】")
        print(input_text)
        print("-" * 80)
        print(f"【token 统计】长度 = {len(token_ids)} tokens")
        print(f"【前 30 个 token_id】 {token_ids[:30].tolist()}")
        print(f"【前 30 个 token 解码】 {tokenizer.convert_ids_to_tokens(token_ids[:30].tolist())}")
        print()


if __name__ == "__main__":
    main()

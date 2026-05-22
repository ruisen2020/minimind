import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    # 根据加载路径初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 如果加载路径是'model'，则使用MiniMindForCausalLM加载原生torch权重，否则使用AutoModelForCausalLM加载transformers格式的权重
    if 'model' in args.load_from:
        # 这里根据输入的参数构建MiniMindConfig对象，并使用该配置初始化MiniMindForCausalLM模型
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        # 根据输入的参数构建权重文件路径，并加载权重到模型中
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        # 加载权重到模型中
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        # 如果输入的lora权重不为'None'，则应用lora模块并加载lora权重到模型中
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    # 返回模型和分词器，并将模型设置为评估模式并移动到指定设备上
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # TextStreamer用于流式输出，可以实时看到模型生成的文本
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 如果输入模式为0，则使用自动测试，否则使用手动输入
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        # setup_seed用于设置随机数种子，保证每次运行的结果一致
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        # 如果是自动测试模式，打印提示语和问题
        if input_mode == 0: print(f'💬: {prompt}')
        # 如果历史对话轮数不为0，则截取最后args.historys轮对话
        conversation = conversation[-args.historys:] if args.historys else []
        # 将当前问题添加到对话中，角色为"user"
        conversation.append({"role": "user", "content": prompt})
        # 根据模型类型，选择是否使用聊天模板进行输入构造
        # 如果是reason模型，则添加思考提示
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
        # 如果不是预训练模型，则使用聊天模板进行输入构造，否则直接使用prompt
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        # 将输入转换为tensor，返回输入ids和attention mask
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        # 生成输出，使用generate函数，返回生成的ids
        # # GenerationMixin.generate 内部逻辑（简化版）
        # while not 停止条件:
        #     # ① 调用你的模型（MiniMindForCausalLM.forward）
        #     outputs = self(input_ids, attention_mask, past_key_values, use_cache=True)
        #     next_token_logits = outputs.logits[:, -1, :]
        #
        #     # ② 应用温度
        #     # 通过温度控制随机性
        #     比如 temperature=0.5 会使概率分布更陡峭，更倾向于选择概率最高的几个token；temperature=1.0 则不改变概率分布；temperature=1.5 会使概率分布更平坦，更倾向于选择更多样化的token
        #     next_token_logits = next_token_logits / temperature
        #
        #     # ③ 应用 top_p 过滤
        #     next_token_logits = top_p_filtering(next_token_logits, top_p)
        #
        #     # ④ 从概率分布采样
        #     probs = softmax(next_token_logits)
        #     next_token = torch.multinomial(probs, num_samples=1)
        #
        #     # ⑤ 拼接到序列
        #     input_ids = torch.cat([input_ids, next_token], dim=-1)
        #
        #     # ⑥ 流式推送到 streamer
        #     streamer.put(next_token)
        #
        #     # ⑦ 判断是否遇到 eos_token_id
        #     if next_token == eos_token_id:
        #         break
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        # generated_ids：[batch_size, 输入长度 + 新生成长度]
        # 由于我们只输入一条数据推理，所以batch_size=1
        # 那么就获取generated_ids[0] 就是我们的输入和生成的完整序列
        # 但是我们只需要生成的序列，所以需要去掉输入的部分
        # inputs：[batch_size, 输入长度] 我们的batch_size = 1，所以直接获取inputs["input_ids"][0] 就是我们的输入序列，长度就是输入长度
        # 那么生成的部分就是 generated_ids[0][输入长度:]，也就是从输入长度开始到结尾的部分，也就是生成的内容
        # skip_special_tokens=True 表示跳过特殊token，比如eos_token_id，pad_token_id等，只保留普通token
        # 最后将生成的token ids解码成文本，得到模型的回复
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # 将模型的回复添加到对话中，角色为"assistant"
        conversation.append({"role": "assistant", "content": response})
        # 计算生成速度，生成的token数量除以生成时间，单位是tokens/s，如果args.show_speed为真，则打印生成速度，否则打印空行
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
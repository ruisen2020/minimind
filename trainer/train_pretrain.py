import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 记录开始时间
    start_time = time.time()
    # 遍历数据集，从start_step + 1开始训练
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 数据是从磁盘先加载到CPU内存，然后从CPU内存加载到GPU
        # to(args.device)是将数据从内存加载到GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 获取当前学习率
        # 设置整体学习率，lr是动态变化的，跟Adam优化器合作
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 修改优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #  autocast_ctx是混合精度训练的上下文管理器，用于将float32转换为float16，以节省内存和加速训练
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        # scale是将loss放大，以避免梯度消失
        # 它会将loss乘以一个比较大的数，这样就避免出现梯度下溢问题（梯度消失）
        scaler.scale(loss).backward()

        # 梯度累加
        if (step + 1) % args.accumulation_steps == 0:
            # 将梯度缩放回真实值
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新参数，如果梯度出现了溢出，就会跳过本次更新
            scaler.step(optimizer)
            # 动态调整缩放因子，如果连续多次没出问题，就会增大缩放因子
            # 如果出了问题，就会自动减小缩放因子
            scaler.update()
            # 清空梯度，因为梯度是累加的，所以需要清空梯度
            # 下一次训练时，梯度会重新计算
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 计算当前损失 
            # loss是累加的，所以需要乘以args.accumulation_steps
            current_loss = loss.item() * args.accumulation_steps
            # 计算当前aux_loss 辅助损失，用于MoE等架构中的专家选择损失
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算当前logits_loss，即主任务损失，用于模型的主任务训练
            current_logits_loss = current_loss - current_aux_loss
            # 计算当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算当前eta_min，即当前轮次的训练时间，单位为分钟
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 保存模型
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 保存模型 为啥要eval： 模型在训练时，会自动设置为train模式，但是保存模型时，需要设置为eval模式，否则保存的模型无法用于推理
            # 为啥无法推理： 因为在训练时会开启dropout，用于防止过拟合，但是推理的模型不需要
            # 并且不需要进行梯度计算，节省显存
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 保存模型的路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取原始模型，分布式训练的模型是DistributedDataParallel，所以需要获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型的参数，state_dict是模型的参数字典，包含模型的权重和偏置
            state_dict = raw_model.state_dict()
            # 保存模型，将模型的参数转换为float16，然后保存到CPU内存，最后保存到磁盘
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 同时保存到检查点目录，保存检查点是保存此刻的训练状态，用于恢复训练
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 保存模型后，设置为训练模式，继续训练
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # dist.is_initialized() 如何使用torchrun启动训练，则dist.is_initialized()返回True，自动开启分布式训练
    # 这里使用的是 PyTorch DDP 的分布式训练
    # torchrun --nproc_per_node=4 script.py
    # torchrun 创建 4 个进程
    # 每个进程设置环境变量：
    # - RANK: 进程排名 (0, 1, 2, 3)
    # - WORLD_SIZE: 4
    # - LOCAL_RANK: 本地排名
    # 每个进程独立执行 script.py
    # 调用 init_distributed_mode()
    # dist.is_initialized() 返回 True  # 分布式已初始化
    # 启用 DDP 训练 
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 为啥会有混合精度
    # 1. 降低内存占用：混合精度使用较低的精度（如bfloat16或float16）进行计算，减少了内存占用，适用于大模型训练
    # 2. 加速训练：混合精度可以加速训练过程，尤其是在GPU上，因为GPU对低精度计算有优化支持
    # 3. 保持精度：通过GradScaler，混合精度训练可以保持与float32相似的精度，同时加速训练
    # 所以会在训练时在精度要求不高的部分使用低精度计算，在需要高精度的部分使用高精度计算，从而达到加速训练和节省内存的目的
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    # 如果使用wandb且是主进程，则初始化wandb
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 初始化预训练数据集，用于训练模型
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果使用了分布式训练，则使用DistributedSampler来采样数据，否则使用普通采样器
    # 每个卡都只获得数据集的一部分，而不是全部数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler用于混合精度训练，可以加速训练过程，但会占用更多的显存
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW优化器，用于更新模型参数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    # 当使用了分布式训练后，每个卡获取到的数据集都不一样，所以需要使用DistributedDataParallel来包模型
    if dist.is_initialized():
        # 告诉 DDP 忽略特定的参数和缓冲区，因为这些参数和缓冲区是不需要同步的
        # 这两个参数是RoPE的参数，RoPE是用于位置编码的，不需要同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 将普通模型转换为分布式数据并行模型，device_ids=[local_rank]表示只使用当前卡
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 训练 args.epochs + 1轮，从0开始
    for epoch in range(start_epoch, args.epochs):
        # 如果使用了分布式训练，则设置采样器的epoch，否则不设置
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个epoch都重新打乱数据集，保证每个epoch的数据集都是随机的
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是第一个epoch且start_step>0，则跳过前start_step个step，否则不跳过
        # 这里是因为checkpoint已经训练过一部分了，所以需要跳过已经训练过的step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 如果使用了分布式训练，则使用DistributedSampler来采样数据，否则使用普通采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器，用于加载数据
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # 如果使用了分布式训练，则销毁分布式进程组
    if dist.is_initialized(): dist.destroy_process_group()
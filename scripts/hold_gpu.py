"""
GPU 占用脚本
===================
用途：在训练任务之间的"空档期"占着 GPU，防止被别人抢走。

特性：
1. 可配置每张卡占用的显存大小（GB）
2. 可配置算力占用强度（通过 matmul 循环维持 GPU Util）
3. 支持多卡（--gpus 0,1,2,3）
4. Ctrl+C 优雅退出，自动释放显存
5. 定时打印心跳日志，方便确认脚本还活着

用法示例：
    # 占用 0 号卡，分配 20GB 显存，低强度算力（默认）
    python hold_gpu.py --gpus 0 --mem_gb 20

    # 占用 0,1 两张卡，每张 40GB 显存，高强度算力（GPU-Util 接近 100%）
    python hold_gpu.py --gpus 0,1 --mem_gb 40 --intensity high

    # 只占显存，不跑算力（GPU-Util ≈ 0%，最省电）
    python hold_gpu.py --gpus 0 --mem_gb 20 --intensity idle
"""
import os
import sys
import time
import signal
import argparse
import threading
from datetime import datetime

import torch


# 全局退出标志，Ctrl+C 时被设为 True
STOP_FLAG = False


def log(msg: str):
    """统一带时间戳的日志"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def signal_handler(signum, frame):
    """Ctrl+C 时触发，让各线程自己退出并释放显存"""
    global STOP_FLAG
    log(f"⚠️  收到信号 {signum}，准备退出...")
    STOP_FLAG = True


def allocate_memory(device: torch.device, mem_gb: float):
    """
    在指定显卡上分配一块指定大小的显存。
    分多个 chunk 分配，避免一次性申请导致碎片化失败。
    """
    bytes_total = int(mem_gb * 1024 ** 3)
    # 每个 chunk 1GB
    chunk_bytes = 1024 ** 3
    # float32: 4 byte/element
    elements_per_chunk = chunk_bytes // 4

    buffers = []
    allocated = 0
    while allocated < bytes_total and not STOP_FLAG:
        remain = bytes_total - allocated
        this_chunk = min(chunk_bytes, remain)
        this_elements = this_chunk // 4
        try:
            buf = torch.empty(this_elements, dtype=torch.float32, device=device)
            # 写入一个数，确保真实分配（而不是 lazy allocation）
            buf.fill_(1.0)
            buffers.append(buf)
            allocated += this_chunk
        except RuntimeError as e:
            log(f"❌ 在 {device} 上分配显存失败（已分配 {allocated / 1024 ** 3:.2f} GB）：{e}")
            break
    log(f"✅ {device} 已分配 {allocated / 1024 ** 3:.2f} GB 显存，共 {len(buffers)} 个 chunk")
    return buffers


def worker(gpu_id: int, mem_gb: float, intensity: str, matrix_size: int):
    """
    单卡 worker：分配显存 + 持续做 matmul（可选）
    intensity:
        - idle : 只占显存，不做计算（GPU-Util ≈ 0%）
        - low  : 低强度计算（GPU-Util 10~30%）
        - high : 高强度计算（GPU-Util 接近 100%）
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    log(f"🚀 Worker 启动：GPU {gpu_id}，目标显存 {mem_gb} GB，算力强度 {intensity}")

    # 1) 先占显存
    buffers = allocate_memory(device, mem_gb)

    # 2) 根据强度选择算力模式
    if intensity == "idle":
        # 纯占显存，循环 sleep
        step = 0
        while not STOP_FLAG:
            time.sleep(30)
            step += 1
            if step % 2 == 0:  # 每分钟一条心跳
                mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
                log(f"💤 GPU {gpu_id} idle 保活中 | reserved={mem_reserved:.2f}GB")
    else:
        # 准备两个矩阵做 matmul
        # low 模式每次算完 sleep 一下，降低占用率
        sleep_sec = 0.0 if intensity == "high" else 0.2
        try:
            a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
            b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        except RuntimeError as e:
            log(f"❌ GPU {gpu_id} 无法分配 matmul 矩阵：{e}，退化为 idle 模式")
            while not STOP_FLAG:
                time.sleep(30)
            return

        step = 0
        last_log_time = time.time()
        while not STOP_FLAG:
            c = torch.matmul(a, b)
            # 防止编译器优化掉结果
            a = c * 0.9999 + a * 0.0001
            torch.cuda.synchronize(device)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

            step += 1
            # 每 30 秒打一条心跳
            if time.time() - last_log_time > 30:
                mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
                log(f"🔥 GPU {gpu_id} 活跃中 | step={step} | reserved={mem_reserved:.2f}GB")
                last_log_time = time.time()

    # 3) 退出前释放显存
    log(f"🧹 GPU {gpu_id} 正在释放显存...")
    del buffers
    torch.cuda.empty_cache()
    log(f"👋 GPU {gpu_id} Worker 退出")


def main():
    parser = argparse.ArgumentParser(description="GPU 占用保活脚本")
    parser.add_argument("--gpus", type=str, default="0",
                        help="要占用的 GPU ID 列表，逗号分隔。例如 '0' 或 '0,1,2,3'")
    parser.add_argument("--mem_gb", type=float, default=10,
                        help="每张卡要占的显存大小（GB）")
    parser.add_argument("--intensity", type=str, default="low",
                        choices=["idle", "low", "high"],
                        help="算力强度：idle=只占显存, low=轻度计算, high=满负荷")
    parser.add_argument("--matrix_size", type=int, default=4096,
                        help="matmul 矩阵大小（high 模式下建议 8192+）")
    args = parser.parse_args()

    # 注册 Ctrl+C 和 kill 信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        log("❌ CUDA 不可用，退出")
        sys.exit(1)

    total_gpus = torch.cuda.device_count()
    log(f"🖥  检测到 {total_gpus} 张 GPU")

    # 解析 gpu 列表
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip() != ""]
    for g in gpu_ids:
        if g < 0 or g >= total_gpus:
            log(f"❌ GPU {g} 不存在（共 {total_gpus} 张卡）")
            sys.exit(1)

    log(f"📋 任务参数：gpus={gpu_ids}, mem_per_gpu={args.mem_gb}GB, intensity={args.intensity}")

    # 每张卡一个线程
    threads = []
    for gid in gpu_ids:
        t = threading.Thread(
            target=worker,
            args=(gid, args.mem_gb, args.intensity, args.matrix_size),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # 主线程等待所有 worker
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        # 双重保险
        global STOP_FLAG
        STOP_FLAG = True

    # 给子线程最多 10 秒清理时间
    for t in threads:
        t.join(timeout=10)

    log("✅ 所有 worker 已退出，主进程结束")


if __name__ == "__main__":
    main()

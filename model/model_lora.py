import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    # 遍历模型的所有模块，找到所有线性层，添加LoRA模块
    for name, module in model.named_modules():
        # 只对全连接层添加LoRA，且只对输入和输出维度相同的层添加LoRA
        # 为啥要输入和输出维度相同的层添加LoRA？
        # 因为输入和输出维度相同的层，是模型的主干，对模型的输出影响最大，所以对这些层添加LoRA，可以有效提升模型的性能
        # 不是相同的维度的层是什么： 比如embedding层，输入是token id，输出是token的embedding，输入和输出维度不相同，所以不对embedding层添加LoRA
        # 那也可以加吧，但是没有意义，因为输入和输出维度不相同的层，对模型的输出影响很小，所以对这些层添加LoRA，对模型的性能提升不大
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 把lora模块添加到module中，新增一个lora属性
            # 那么原始模块不就改名字了吗？
            # 不会，只是在module上新增了一个lora属性，不影响原始模块的forward函数
            # 具体来说：它会将lora的参数都设置为lora的名字
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            # 这里定义了forward_with_lora函数，这个函数的作用是：先执行原始的forward函数，然后执行lora的forward函数，最后将两个结果相加，得到最终的结果
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            # 重写forward函数
            module.forward = forward_with_lora

# 加载lora参数
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    # 遍历模型的所有模块，找到所有lora模块，加载lora参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)

# 保存lora参数
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)

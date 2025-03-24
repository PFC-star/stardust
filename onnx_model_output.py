import torch
import torch.nn as nn

# 钩子函数，用于打印每一层的输出尺寸
def hook_fn(module, input, output):
    # 过滤掉 nn.Sequential 层
    if not isinstance(output, tuple):
        print(f"{module.__class__.__name__} : {output.shape}")

# 定义一个简单的模型，具有多个输出
class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 10)  # 第一个输出
        self.fc2 = nn.Linear(32 * 56 * 56, 5)   # 第二个输出

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        out1 = self.fc1(x)  # 第一个输出
        out2 = self.fc2(x)  # 第二个输出
        return out1, out2  # 返回多个输出

# 实例化模型
model = MultiOutputModel()

# 注册钩子函数，只对非 Sequential 层注册
for layer in model.modules():
    #print(layer)
    if not isinstance(layer, nn.Sequential):  # 跳过 nn.Sequential 层
        layer.register_forward_hook(hook_fn)

model.eval()

# 创建示例输入，形状为 (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)

# 导出为 ONNX，处理多个输出
torch.onnx.export(
    model,                        # PyTorch 模型
    dummy_input,                  # 示例输入
    "multi_output_model.onnx",    # ONNX 文件保存路径
    export_params=True,           # 是否导出模型参数
    opset_version=11,             # ONNX 的 opset 版本
    do_constant_folding=True,     # 是否进行常量折叠优化
    input_names=['input'],        # 输入的名称
    output_names=['output1', 'output2'],  # 多个输出的名称
    dynamic_axes={                 # 指定动态维度
        'input': {0: 'batch_size'},         # 输入的 batch_size 维度是动态的
        'output1': {0: 'batch_size'},       # 第一个输出的 batch_size 维度是动态的
        'output2': {0: 'batch_size'}        # 第二个输出的 batch_size 维度是动态的
    }
)

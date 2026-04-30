import torch
import torch.nn as nn
import torch.profiler


# 1. 随便定义一个模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])
        x = self.linear(x)
        return x


model = Model().cuda()
x = torch.randn(16, 3, 224, 224).cuda()

# 2. 🔥 PyTorch Profiler 核心代码
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # 看GPU必须加
    ],
    record_shapes=True,  # 记录张量形状
    profile_memory=True,  # 记录内存
    with_stack=True,  # 记录代码行
) as prof:
    # 🔥 要 profiling 的代码
    for _ in range(5):
        y = model(x)
        y.sum().backward()
        prof.step()  # 标记 step

# 3. 输出结果
print("=== 算子耗时排名 ===")
print(
    prof.key_averages().table(
        sort_by="cuda_time_total",  # 按GPU耗时排序
        row_limit=15,
    )
)

# 4. 导出 Chrome 可视化（超级好用）
prof.export_chrome_trace("trace.json")

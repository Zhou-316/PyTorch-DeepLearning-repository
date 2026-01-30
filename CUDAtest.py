import torch
print("CUDA available:", torch.cuda.is_available())    ##检查CUDA是否可用
print("GPU count:", torch.cuda.device_count())         ##检查GPU数量
if torch.cuda.is_available():                         
   print("GPU name:", torch.cuda.get_device_name(0))   ##获取GPU名称


print("PyTorch 编译时支持的 GPU 架构:")
print(torch.cuda.get_arch_list())                      ##查看PyTorch编译时支持的GPU架构列表（回答以sm_形式，即为支持的架构）

print("PyTorch version:", torch.__version__)           #查看PyTorch版本
print("CUDA version used to build PyTorch:", torch.version.cuda)     #查看PyTorch编译时使用的CUDA版本
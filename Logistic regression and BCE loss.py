#import torchvision
##train代表是训练集还是测试集，download自动下载
##MNIST数据集包含手写数字0-9的灰度图像，是常用的机器学习数据集
#train_dataset = torchvision.datasets.MNIST(root='./data',train=True,download=True)
#test_dataset = torchvision.datasets.MNIST(root='./data',train=False,download=True)

import torch
import torch.nn.functional as F  #导入函数接口
import math
x_data =torch.tensor([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0]])
y_data =torch.tensor([[0.0],[0.0],[0.0],[0.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]])

# Logistic回归模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  #输入维度1，输出维度1

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  #使用sigmoid函数将logits转换为概率
        return y_pred
model = LogisticRegressionModel()
# 二元交叉熵损失函数(Binary二元的 Cross Entropy Loss,BCE Loss)
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses=[]
# 训练模型
for epoch in range(5000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    losses.append(loss.item())
    print("epoch", epoch, "loss:", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#绘图1:训练损失曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5)) 
plt.plot(range(1, len(losses) + 1), losses, 'b', label='Training loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘图2:预测结果
import numpy as np
x=np.linspace(0, 9, 100)   #生成0到9之间的100个点
x_t=torch.tensor(x,dtype=torch.float32).view((100,1))  #转换为100行1列的张量
y_t=model(x_t)
y=y_t.data.numpy()  #转换为numpy数组
y=np.clip(y, 0, 1)  #将概率限制在0到1之间
plt.plot(x, y)
plt.xlabel('Hours')
plt.ylabel('Pass Probability')
plt.title('Logistic Regression Prediction')
plt.grid()
plt.show()

#说明：这还是一个一维的回归算法，接下来会做多维的，包括根据MINST数据集的训练
import torch
import torch.nn.functional as F  #导入函数接口
import math
import numpy as np
xy=np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)  #加载数据集
x_data=torch.from_numpy(xy[:,:-1])  #提取输入特征
y_data=torch.from_numpy(xy[:,[-1]])  #提取标签      #torch.from_numpy()将numpy数组转换为torch张量

class MultiDimInputModel(torch.nn.Module):
    def __init__(self):
        super(MultiDimInputModel, self).__init__()
        self.linear = torch.nn.Linear(8, 6)  # 定义一个线性层
        self.linear2 = torch.nn.Linear(6, 4)  # 定义第二个线性层
        self.linear3 = torch.nn.Linear(4, 1)  # 定义第三个线性层
        self.activate = torch.nn.ReLU()  #在这里改激活函数
    def forward(self, x):
        x=self.activate(self.linear(x))  #通过第一个线性层
        x=self.activate(self.linear2(x))  #通过第二个线性层
        x=torch.sigmoid(self.linear3(x))  #通过第三个线性层
        return x
model=MultiDimInputModel()  #实例化模型
criterion=torch.nn.BCELoss(size_average=True)  #仍然使用BCE损失，因为输出还是两个类别
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)  #优化器
losses=[]
for epoch in range(1000):
    y_pred=model(x_data)  
    loss=criterion(y_pred,y_data)  
    #每十轮打印一次损失
    if epoch % 10 == 0:
        losses.append(loss.item())
        print(f'Epoch {epoch+1}/1000 Loss: {loss.item():.4f}')
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5)) 
plt.plot(range(1, len(losses) + 1), losses, 'b', label='Training loss')
plt.title('Training Loss per ten Epoch')
plt.xlabel('Epochs(*10)')
plt.ylabel('Loss')
plt.legend()
plt.show()

 #注意：还没有做mini-batch处理，整个数据集一次性输入模型进行训练。后续会做。
 #此外，可以多试一试别的激活函数
 #在使用ReLU等，为了避免输出0或1，在最后一层使用sigmoid函数将输出限制在0到1之间
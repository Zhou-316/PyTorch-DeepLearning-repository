import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ##启用GPU，CUDA加速
print("Using device:", device)
#注：为什么这里先用了CUDA而不是Mini-batch来加速呢？因为我想用hh，dataloader的用法后面才会学到
# 加载数据集
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
x_data = x_data.to(device)
y_data = y_data.to(device)

class MultiDimInputModel(torch.nn.Module):
    def __init__(self, activation):
        super(MultiDimInputModel, self).__init__()
        self.linear = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = activation          ##################
    
    def forward(self, x):
        x = self.activate(self.linear(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))  # 输出层保持sigmoid不变,为了防止输出0或1
        return x

activations = [torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.ReLU()]
names = ['Sigmoid', 'Tanh', 'ReLU']
results = {name: [] for name in names}

for act_name, activation in zip(names, activations):
    for run in range(10):  # 对每种激活函数进行10次实验
        model = MultiDimInputModel(activation=activation).to(device)     #前一个activate是类的属性，后一个是传入的激活函数
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        losses = []
        for epoch in range(5000):
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            if epoch % 10 == 0:
                losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        results[act_name].append(losses)

plt.figure(figsize=(10, 6))

for act_name in names:
    losses_array = np.array(results[act_name])  # shape: (runs, epochs/10)
    mean_loss = losses_array.mean(axis=0)
    std_loss = losses_array.std(axis=0)
    
    plt.plot(range(1, len(mean_loss) + 1), mean_loss, label=act_name)
    plt.fill_between(range(1, len(mean_loss) + 1), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)

plt.xlabel("Epochs(*10)")
plt.ylabel("Loss")
plt.title("Training Loss per Ten Epochs for Different Activation Functions")
plt.legend()
plt.grid(True)
plt.show()
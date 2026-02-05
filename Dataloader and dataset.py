import torch    
from torch.utils.data import Dataset,DataLoader       #Dataset是一个抽象类，只能被继承
import numpy as np
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):          ##自定义数据集类,括号内代表继承Dataset类
    def __init__(self,filepath):
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(data[:, :-1])  ##特征数据，前八列
        self.y_data = torch.from_numpy(data[:,[-1]]).float()  ##标签数据，最后一列
        self.len = data.shape[0]             

    def __len__(self):                   ##返回数据集的大小
        return self.len

    def __getitem__(self, idx):          ##使得数据集可以通过索引访问
        return self.x_data[idx], self.y_data[idx]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(8, 6)  # 定义一个线性层
        self.linear2 = torch.nn.Linear(6, 4)  # 定义第二个线性层
        self.linear3 = torch.nn.Linear(4, 2)  # 定义第三个线性层
        self.linear4 = torch.nn.Linear(2, 1)  # 定义第四个线性层
        self.activate = torch.nn.ReLU()  #在这里改激活函数
    def forward(self, x):
        x=self.activate(self.linear(x))  #通过第一个线性层
        x=self.activate(self.linear2(x))  #通过第二个线性层
        x=self.activate(self.linear3(x))  #通过第三个线性层
        x=torch.sigmoid(self.linear4(x))  #通过第四个线性层
        return x
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ##启用GPU，CUDA加速
    dataset=DiabetesDataset('diabetes.csv')  #实例化自定义数据集，括号内传入数据文件路径
    train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True) #因为Dataloader也使用了多线程，所以要把它放在主函数里
    model=Model().to(device)  #实例化模型并移动到GPU
    criterion=torch.nn.BCELoss(reduction='mean')  
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)  
    losses=[]
    for epoch in range(3000):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader,0):      #i代表第几个batch
            inputs, labels = inputs.to(device), labels.to(device)  #将数据移动到GPU
            y_pred=model(inputs)  
            loss=criterion(y_pred,labels)  
            epoch_loss += loss.item()
            #每十轮打印一次损失
            if epoch % 10 == 0 and i == len(train_loader) - 1:
                avg_loss = epoch_loss / len(train_loader)
                losses.append(avg_loss)
                print(f'Epoch {epoch+1}/3000 Loss: {avg_loss:.4f}')
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b', label='Training loss')
    plt.title('Training Loss per ten Epoch')
    plt.xlabel('Epochs(*10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("Using device:", device)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))])
class MNISTTrainer(torch.nn.Module):
    def __init__(self):
        super(MNISTTrainer,self).__init__()
        self.conv1 =torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 =torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(320,50) #卷积层输出的特征图大小为4x4，(为什么？自己算）通道数为20，所以全连接层输入的特征数为20*4*4=320
        self.fc2 = torch.nn.Linear(50,10)
    def forward(self,x):
        x=torch.relu(self.pooling(self.conv1(x))) ##卷积层1 + 激活函数 + 池化层
        x=torch.relu(self.pooling(self.conv2(x))) ##卷积层2 + 激活函数 + 池化层
        x=x.view(-1,320) ##将特征图展平为一维向量，准备输入全连接层，这步也叫flatten
        x=torch.relu(self.fc1(x)) ##全连接层1 + 激活函数
        x=self.fc2(x) ##全连接层2，后面会使用CrossEntropyLoss，它内部会应用Softmax函数
        return x
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=transform,download=True) #train参数指定了是加载训练集还是测试集，不必担心
train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
model=MNISTTrainer().to(device)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
def train(epoch):
    model.train()
    running_loss=0.0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx % 100 == 99: ##每100个批次打印一次平均损失
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
            running_loss=0.0
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            _,predicted=torch.max(output.data,1)
            total+=target.size(0) ##统计batch总样本数
            correct+=(predicted==target).sum().item()
    print(f'Accuracy: {100*correct/total:.2f}%')

if __name__ =="__main__":
    num_epochs=10
    for epoch in range(num_epochs):
        train(epoch)
        test()
    



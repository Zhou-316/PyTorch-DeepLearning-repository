import torch    
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ##启用GPU，CUDA加速
print("Using device:", device)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))]) ##数据预处理：将图像转换为PyTorch张量
#   这行代码的作用是对MNIST数据集进行归一化处理,0.1307和0.3081分别是MNIST数据集的平均值和标准差
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=transform,download=True)
##知识点1：
#   datasets.MNIST是torchvision库中提供的一个用于加载MNIST数据集的类
#   root参数指定数据集存储的路径
#   train参数指定是加载训练集还是测试集
#   transform参数用于对数据进行预处理，这里使用transforms.ToTensor()将图像转换为PyTorch张量
#   download参数指定如果数据集不存在，是否从网上下载
train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
##知识点2：
#   DataLoader是PyTorch中用于加载数据的一个类
#   dataset参数传入数据集对象
#   batch_size参数指定每个批次加载多少样本
#   shuffle参数指定是否在每个epoch开始时打乱数据
#   num_workers并行加载数据的线程数，由CPU核心数决定，一般设置为4或8，注意我们虽然使用了CUDA加速，但数据加载仍然是CPU的任务

#   pin_memory=True的含义和原理：普通内存（Pageable Memory）和锁页内存（Pinned Memory）
#   Pageable Memory是操作系统可以随时将其内容交换到磁盘上的内存（称为SWap，虚拟内存），而Pinned Memory则是锁定在物理内存中的，不能被交换到磁盘
#   如果不启用，PyTorch 必须先在后台悄悄申请一块“锁页内存”，然后把普通内存中的数据先拷贝到这块锁页内存中，最后再从锁页内存拷贝到 GPU 显存中
#   启用 pin_memory 后，DataLoader 会直接在锁页内存中分配张量，GPU通过DMA技术直接访问。这样就省去了中间拷贝的步骤，提高了数据传输效率。

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=torch.nn.Linear(28*28,512) 
        self.fc2=torch.nn.Linear(512,256)
        self.fc3=torch.nn.Linear(256,128)
        self.fc4=torch.nn.Linear(128,64)
        self.fc5=torch.nn.Linear(64,10)
    def forward(self,x):
        x=x.view(-1,28*28) ##将输入图像展平为一维向量
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        x=torch.relu(self.fc4(x))
        x=self.fc5(x) ##输出层不使用激活函数，因为后面会使用CrossEntropyLoss，它内部会应用Softmax函数
        return x
#if __name__ == "__main__": 算了不要了，反正也不需要多线程。
model=Net().to(device) 
criterion=torch.nn.CrossEntropyLoss() 
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#momentum参数的作用是加速SGD优化器的收敛速度，减少震荡。它通过在更新参数时考虑之前的更新方向来实现这一点。
#具体来说，momentum会在每次更新时将当前的梯度与之前的更新方向进行加权平均，从而使得优化器在下降过程中更稳定，避免陷入局部最小值。                
def train(epoch):
    model.train()                                 
    running_loss=0.0 
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device) ##将数据和标签移动到GPU上进行计算
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
    with torch.no_grad(): ##在测试阶段不需要构建计算图和计算梯度，节省内存和计算资源
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            _,predicted=torch.max(output.data,1) ##获取预测结果
            #torch.max()函数返回两个值，分别是最大值和索引，这里我们只关系索引
            total+=target.size(0) ##统计batch总样本数
            correct+=(predicted==target).sum().item() ##统计正确预测的样本数
            #predicted == target 会逐元素比较，得到一个布尔张量，.sum() 会把 True 当作 1，False 当作 0，所以如果 32 张中有 28 张预测对了，.sum() 就是 28
    print(f'Accuracy: {100*correct/total:.2f}%')

if __name__ == "__main__":
    num_epochs=10
    for epoch in range(num_epochs):
        train(epoch)
        test()


import torch
import matplotlib.pyplot as plt
# Mini-batch Data
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

#用pytorch搭建模型，重要的是构造计算图
class LinearModel(torch.nn.Module):          ##python拾遗：LinearModel继承torch.nn.Module，是子类
    def __init__(self):                      ##初始化函数，定义模型的层
        super(LinearModel, self).__init__()  ##调用父类的初始化函数，这条比较公式化，意思是既然LinearModel是子类的话也必须经过父辈的初始化
        self.linear = torch.nn.Linear(1, 1)  #输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)               ##前馈
                                            ##对象后面加括号表示调用该对象的__call__函数，而__call__函数中会调用forward函数
#知识点补充1：
#   关于__call__函数的说明：
#   在Python中，__call__是一个特殊的方法，它允许类的实例像函数一样被调用。
#   当你对一个对象使用括号（例如 obj()）时，Python会自动调用该对象的 __call__ 方法。
#   例如：
#   class MyClass:
#   def __init__(self):
#       pass
#   def __call__(self, *args, **kwargs):
#       print("Hello"+str(args))
# obj = MyClass()
# obj(1, 2, 3)  # 这将输出 "Hello(1, 2, 3)"
# obj(1,2,3,x=4,y=5)  # 这将输出 "Hello(1, 2, 3)"
#也就是说，args是位置参数的元组，kwargs是关键字参数的字典。当你不知道会传入多少参数时，args会帮你储存，这种方式非常有用。
model = LinearModel()                      ##实例化模型
criterion = torch.nn.MSELoss(size_average=False)            ##均方误差损失函数
#知识点补充2：
#   关于均方误差损失函数MSELoss的说明：这也是一个类，在nn.Module中定义的
#   criterion作为对象需要两个参数，y_pred和y_true
#   size_average=False表示不对损失取平均值，而是返回总和，影响不大
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  ##随机梯度下降优化器，传入模型的参数和学习率
#知识点补充3：
#   采用SGD算法进行权重的更新，这点无需多言
#   model.parameters()会返回模型中所有需要优化的参数，这里就是线性层的权重和偏置
#   model.parameters()是nn.Module类中的一个返回一个生成器（generator），里面包含模型中所有 nn.Parameter 类型的张量（即需要通过反向传播更新的权重和偏置）。
#   事实上，存在不同算法的优化器，比如Adam、RMSprop等
losses=[]
#   下面正式训练，四个过程：前馈、计算损失、反向传播、更新参数   
for epoch in range(800):
    y_pred=model(x)
    loss=criterion(y_pred, y)               ##计算损失
    losses.append(loss.item())              ##记录损失值用于画图
    print("epoch", epoch, "loss:", loss.item())
    optimizer.zero_grad()                   ##清零梯度
    loss.backward()                         ##反向传播计算梯度
    optimizer.step()                        ##更新参数
print("w:", model.linear.weight.item(), "b:", model.linear.bias.item())
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("prediction after training:", 4, y_test.item())

#画图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses, 'b', label='Training loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
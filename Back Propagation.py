import torch

# Data
xs = torch.tensor([1.0, 2.0, 3.0])
ys = torch.tensor([2.0, 4.0, 6.0])

# Parameters (0-dim tensors)
w = torch.tensor(1.0, requires_grad=True)
t = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

def forward(x):  ## Define the model
    return x.pow(2) * w + t * x + b

def loss_fn(x, y):   ## Define the loss function
    y_pred = forward(x)
    return (y_pred - y).pow(2)

print("prediction before training:", 4, forward(torch.tensor(4.0)).item())

lr = 0.01   # learning rate学习率
for epoch in range(800):
    epoch_loss = 0.0      #每一轮训练的总损失
    for xi, yi in zip(xs, ys):
        l = loss_fn(xi, yi)
        l.backward()      #反向传播计算梯度
        with torch.no_grad():
            w -= lr * w.grad
            t -= lr * t.grad
            b -= lr * b.grad
        w.grad.zero_()   #注意清零梯度，否则梯度会累加
        t.grad.zero_()
        b.grad.zero_()
        epoch_loss += l.item()
    if epoch % 10 == 0:   ##每十轮打印一次损失，方便观察训练情况
        print("epoch", epoch, "avg loss:", epoch_loss / len(xs))
print("w:", w.item(), "t:", t.item(), "b:", b.item())
print("prediction after training:", 4, forward(torch.tensor(4.0)).item())
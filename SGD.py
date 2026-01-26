import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]
b = 0.0
w = 1.0

def forward(x):
    return x * w+b       # simple linear function

def loss(x, y):          #损失函数
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):              #权重梯度下降算法
    return 2 * x * (x * w+b - y)

def gradient_b(x, y):
    return 2 * (x * w + b - y)    #偏置项的梯度,注意与上面不同（可以推导）

epoch_list = []
loss_list = []

print('Predict (before training)', 4, forward(4))

for epoch in range(100):            #训练
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        b = b - 0.01 * gradient_b(x, y)
        print("\tgrad: ", x, y)
        l = loss(x, y)
        epoch_list.append(epoch)
        loss_list.append(l)

    print("progress:", epoch, "w=", w, "b=",b,"loss=", l)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, loss_list)  ##绘图
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
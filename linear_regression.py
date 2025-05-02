import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu" #设置设备

def Timer(func):#计时器
    def wrapper(*args, ** kwargs):
        start_time = time.perf_counter()
        result = func(*args, ** kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} 耗时: {end_time - start_time:.6f} 秒")
        return result

    return wrapper


@Timer
def load_data():#从linear_data.csv读取数据
    data = pd.read_csv("linear_data.csv")
    x = torch.tensor(data["x"].values, dtype=torch.float32).reshape(-1, 1).to(device)
    y = torch.tensor(data["y"].values, dtype=torch.float32).reshape(-1, 1).to(device)
    return x, y


x, y = load_data()

train_split = int(len(x) * 0.8)#分割数据，这里设置比例为8：2
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]


class LinearRegression(nn.Module):#线性回归模型
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression().to(device)
loss = nn.MSELoss()#选用平方均差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)#设置学习率为0.01
epochs = 2000#设置训练轮次为2000
losses = []


@Timer
def train_model(model, x_train, y_train, epochs):#训练模型
    model.train()
    for epoch in range(epochs):
        outputs = model(x_train)
        l1 = loss(y_train, outputs)
        optimizer.zero_grad()
        l1.backward()
        optimizer.step()
        losses.append(l1.item())
        if (epoch + 1) % (epochs // 10) == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {l1.item():.4f}')
    return model


model = train_model(model, x_train, y_train, epochs)


@Timer
def evaluate(model, x_test, y_test):#模型评估，返回损失函数的值
    model.eval()  # 切换到评估模式
    with torch.inference_mode():
        outputs = model(x_test)
        mse = loss(outputs, y_test)
    return mse


mse = evaluate(model, x_test, y_test)
print(f"Test MSE: {mse.item():.4f}")

print(f"x:{model.linear.weight.item()},y:{model.linear.bias.item()}")

@Timer
def visualize_results(x, y, model, losses):#可视化显示
    model.eval()  # 确保模型处于评估模式
    plt.figure(figsize=(16, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-o', markersize=4, linewidth=1)
    plt.title('Training Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    # 拟合结果（使用 inference_mode 加速）
    with torch.inference_mode():  # 高性能推理模式
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        predicted = model(x).cpu()  # 无需 detach()，inference_mode 已处理

    plt.subplot(1, 2, 2)
    plt.scatter(x_cpu, y_cpu, alpha=0.3, label='Data')
    plt.plot(x_cpu, predicted, 'r-', linewidth=2, label='Fitted Line')
    plt.title('Regression Fit', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


visualize_results(x_train, y_train, model, losses)

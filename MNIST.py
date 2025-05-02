import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import requests
from pathlib import Path
import time
import random
def Timer(func):#计时器
    def wrapper(*args, ** kwargs):
        start_time = time.perf_counter()
        result = func(*args,  ** kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} 耗时: {end_time - start_time:.6f} 秒")
        return result

    return wrapper
#读取训练集和测试集
train_data = datasets.MNIST(root="data",train = True,download = True,transform = ToTensor())
test_data = datasets.MNIST(root="data",train= False,download = True,transform = ToTensor())
#选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = train_data.classes#分类的种类
#batch_size
BATCH_SIZE =32
#设置训练和测试的dataloader
train_dataloader = DataLoader(train_data,batch_size = BATCH_SIZE,shuffle = True)
test_dataloader = DataLoader(test_data,batch_size = BATCH_SIZE, shuffle = False)

#下载helper_functions.py使用里面的accuracy_fn
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
from helper_functions import accuracy_fn
#训练模型
def train_step(model:nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device = device
              ):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (x,y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc += accuracy_fn(y,y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_acc
#测试模型（准确率）
def test_step(model:nn.Module,
             data_loader:torch.utils.data.DataLoader,
             loss_fn:nn.Module,
             optimizer:torch.optim.Optimizer,
             accuracy_fn,
             device:torch.device = device
             ):
    test_loss, test_acc = 0,0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (x,y) in enumerate(data_loader):
            x, y =x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred,y)
            test_acc += accuracy_fn(y,y_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
#评估模型
def eval_model(model:nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:nn.Module,
               accuracy_fn,
               optimizer:torch.optim.Optimizer,
               device:torch.device = device
              ):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true=y,y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

#定义模型，其中act_fn是激活函数，conv_layers是卷积层的层数
class BaseMNISTModel(nn.Module):
    def __init__(self, input_shape=1, hidden_units=32, output_shape=10, act_fn=nn.ReLU, conv_layers=4):
        super().__init__()
        layers = []
        in_channels = input_shape
        for i in range(conv_layers):
            out_channels = hidden_units if i == 0 else 64
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                act_fn(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels

        self.block = nn.Sequential(*layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape, 28, 28)
            dummy_output = self.block(dummy_input)
            flatten_dim = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, output_shape)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x


model = BaseMNISTModel().to(device)

def visualize_test_predictions(model, test_loader, classes, device, n_images=16):
    torch.manual_seed(42)

    test_features, test_labels = next(iter(test_loader))

    indices = random.sample(range(len(test_features)), n_images)

    plt.figure(figsize=(12, 12), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    model.eval()
    with torch.inference_mode():
        predictions = model(test_features.to(device)).argmax(dim=1).cpu()

    cols = 4
    rows = (n_images + cols - 1) // cols

    for i, idx in enumerate(indices):
        ax = plt.subplot(rows, cols, i + 1)
        img = test_features[idx].squeeze().permute(1, 2, 0) if test_features[0].shape[0] == 3 else test_features[
            idx].squeeze()
        true_label = test_labels[idx].item()
        pred_label = predictions[idx].item()

        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')

        color = "green" if true_label == pred_label else "red"
        title = f"True: {classes[true_label]}\nPred: {classes[pred_label]}"
        ax.set_title(title, color=color, fontsize=9, pad=2)

        if i == 0:  # 仅在第一个子图显示尺寸信息
            print(f"Image size: {test_features[idx].shape}")
            print(f"Label size: {test_labels[idx].shape}")

    plt.suptitle('Test Set Predictions (Green=Correct, Red=Wrong)',
                 y=0.93, fontsize=14, fontweight='bold')
    plt.show()


#模型训练前的可视化显示
visualize_test_predictions(model=model,
                           test_loader=test_dataloader,
                           classes=classes,
                           device=device)

from tqdm import tqdm
loss_fn = nn.CrossEntropyLoss()#损失函数，这里使用交叉熵函数
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)#学习率定为0.1
@Timer
def func():
    epochs = 100
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}:------------')
        acc=train_step(model = model,data_loader = train_dataloader, loss_fn = loss_fn,optimizer = optimizer,accuracy_fn = accuracy_fn,device=device)
        test_step(model = model,data_loader = test_dataloader,loss_fn = loss_fn,optimizer = optimizer,accuracy_fn =accuracy_fn,device = device)
        if acc >= 99.5:#当训练准确率大于等于99.5时暂停
            print(acc)
            break

func()
model_3_results = eval_model(model=model,data_loader = test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,optimizer=optimizer,device=device)
print(model_3_results)#打印评估结果
#模型训练后的可视化显示
visualize_test_predictions(model=model,
                          test_loader=test_dataloader,
                          classes=classes,
                          device=device)

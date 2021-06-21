import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import os
from loss.focal import FocalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 128
EPOCH = 240
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(block, 64,  layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# resnet18
model = ResNet(ResidualBlock, [2, 2, 2, 2])

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# lr=0.1, 0.01, 0.0001
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# Multistep
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [135, 185], gamma=0.1, last_epoch=-1)


transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Augumentation cutout
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


accuracy_stats = {
    'train': [],
    "test": []
}
loss_stats = {
    'train': [],
    "test": []
}

# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    loss_stats['train'].append(train_loss / len(trainloader))
    accuracy_stats['train'].append(correct / total)
    print(f'Epoch {epoch}: | Train Loss: {train_loss / len(trainloader):.5f} | Train Acc: {correct / total:.3f}')

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        loss_stats['test'].append(test_loss / len(testloader))
        accuracy_stats['test'].append(correct / total)
        print(f'Epoch {epoch}: | Test Loss: {test_loss / len(testloader):.5f} | Test Acc: {correct / total:.3f}')

if __name__ == "__main__":
    print("Summary of resnet18")
    summary(model, input_size=(3, 32, 32))
    for epoch in range(EPOCH):
        train(epoch)
        test(epoch)
        scheduler.step()
    torch.save(model.state_dict(), 'resnet18_model.pkl')
    # save result
    np.save('reset18_acc.npy', accuracy_stats)
    np.save('resnet18_loss.npy', loss_stats)
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Test Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Test Loss/Epoch')
    plt.savefig('resnet18_result.png', dpi=300)




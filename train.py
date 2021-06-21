from Alexnet import *
from data_argumentation import *
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






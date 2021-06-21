import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

classes = train_set.classes
# torch.manual_seed(43)
# val_size = 5000
# train_size = len(dataset) - val_size
# train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

def show_stats(set, classes):
    class_count = {}
    for _, index in set:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count

class_train = show_stats(train_set, classes)
# class_val = show_stats(val_set, classes)
class_test = show_stats(test_set, classes)
print("train:", class_train)
print("test:", class_test)

batch_size = 128
train_loader = DataLoader(train_set, batch_size, shuffle=True)
for images, _ in train_loader:
    print("image shape:", images.shape)
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    plt.savefig('cifar_im.png', dpi=300)
    plt.show()
    break

# resnet-
Resnet18 for CIFAR-10 Image Classification with Data Argumentation


Resnet18 and three methods of data argumentation were evaluated on CIFAR-10.Implementation Details: We implemented Resnet using Pytorch deep learning framework. The base learning rate was set as 0.1 and was reduced by multistep policy with gamma of 0.1. The momentum was set as 0.9 and the weight decay was set as 0.0005. The training of models was completed on a 1080ti GPU with 12GB memory. During the training stage, we set the actual batch size as 128 and the maximum epochs as 240 . It took about 6 hours to finish optimizing the model.

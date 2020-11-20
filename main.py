import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import copy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def partial_grad(self, data, target, loss_function):
        outputs = self.forward(data)
        loss = loss_function(outputs, target)
        loss.backward()
        return loss

    def calculate_loss_grad(self, dataset, loss_function, n_samples):
        total_loss = 0.0
        full_grad = 0.0
        for i_grad, data_grad in enumerate(dataset):
            inputs, labels = data_grad
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            total_loss += (1. / n_samples) * self.partial_grad(inputs, labels, loss_function).data.item()

        for para in self.parameters():
            full_grad += para.grad.data.norm(2) ** 2

        return total_loss, (1. / n_samples) * np.sqrt(full_grad.data.item())

    def svrg_training(self, dataset, loss_function, n_epoch, learning_rate):
        total_loss_epoch = [0 for i in range(n_epoch)]
        grad_norm_epoch = [0 for i in range(n_epoch)]
        full_grad = 0.0
        all_losses = []
        all_gradients = []
        for epoch in range(n_epoch):
            running_loss = 0.0

            yt = copy.deepcopy(self)
            ds = copy.deepcopy(self)

            ds.zero_grad()
            total_loss_epoch[epoch], grad_norm_epoch[epoch] = ds.calculate_loss_grad(dataset, loss_function, n_samples)
            print(total_loss_epoch[epoch], grad_norm_epoch[epoch])
            # Run over the dataset
            for i_data, data in enumerate(dataset):
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                yt.zero_grad()
                _ = yt.partial_grad(inputs, labels, loss_function)

                self.zero_grad()
                cur_loss = self.partial_grad(inputs, labels, loss_function)

                # Update the current weights using the gradients from the previous state of the network
                # and the gradient from the current state of the network
                for param1, param2, param3 in zip(self.parameters(), yt.parameters(),
                                                  ds.parameters()):
                    param1.data -= (learning_rate) * (
                            param1.grad.data - param2.grad.data + (1. / n_samples) * param3.grad.data)

                for para in self.parameters():
                    full_grad += para.grad.data.norm(2) ** 2

                running_loss += cur_loss.data.item()
                if i_data % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_data + 1, running_loss / 50))
                    all_losses.append(running_loss / 50)
                    running_loss = 0.0
                    all_gradients.append((1. / 50000) * np.sqrt(full_grad.item()))

        return total_loss_epoch, grad_norm_epoch, all_losses, all_gradients

    def sag_training(self, dataset, loss_function, n_epoch, learning_rate):
        total_loss_epoch = [0 for i in range(n_epoch)]
        grad_norm_epoch = [0 for i in range(n_epoch)]
        all_losses  = []
        all_gradients = []
        n_batches = len(trainloader)
        full_grad = 0.0
        y = {}
        for i in range(n_batches):
            y[i] = []
        d = None

        for epoch in range(n_epoch):
            running_loss = 0.0
            if epoch != 0  and epoch % 2 == 0:
                learning_rate /= 10

            print(total_loss_epoch[epoch], grad_norm_epoch[epoch])
            # Run over the dataset
            for i_data, data in enumerate(dataset):
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                # yt.zero_grad()
                # _ = yt.partial_grad(inputs, labels, loss_function)

                self.zero_grad()
                cur_loss = self.partial_grad(inputs, labels, loss_function)

                if d is None:
                    d = [x.grad.data for x in self.parameters()]
                else:
                    if len(y[i_data]) == 0:
                        d = [y + x.grad.data for x,y in zip(self.parameters(), d)]
                    else:
                        d = [x.grad.data + y + z for x,y,z in zip(self.parameters(), d, y[i_data])]

                y[i_data] = [x.grad.data for x in self.parameters()]

                for param1, param2 in zip(self.parameters(), d):
                    param1.data -= learning_rate / n_batches * param2


                for para in self.parameters():
                    full_grad += para.grad.data.norm(2) ** 2

                running_loss += cur_loss.data.item()
                if i_data % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_data + 1, running_loss / 50))
                    all_losses.append(running_loss / 50)
                    running_loss = 0.0
                    all_gradients.append((1. / 50000) * np.sqrt(full_grad.item()))

        return total_loss_epoch, grad_norm_epoch, all_losses, all_gradients


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    n_epoch = 6
    learning_rate = 0.1
    start = time.time()
    n_samples = len(trainloader)
    print(n_samples)
    # total_loss_epoch, grad_norm_epoch, losses, grads = net.svrg_training(trainloader, criterion, n_epoch, learning_rate)
    total_loss_epoch, grad_norm_epoch, losses, grads = net.sag_training(trainloader, criterion, n_epoch, learning_rate)

    end = time.time()
    print('time is : ', end - start)
    print('Finished Training')

    net.eval()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = net(images.to('cuda'))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.to('cpu') == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    epoch_abs = [i for i in range(len(losses))]
    plt.plot(epoch_abs, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function ')
    plt.title('SAG : Evolution of minimizing the Objective Function')
    plt.grid()
    plt.show()

    epoch_abs = [i for i in range(len(grads))]
    plt.plot(epoch_abs, grads)
    plt.xlabel('iteration')
    plt.ylabel('Norm of the gradients')
    plt.title('SAG : Evolution of the accumulation of normalized gradients')
    plt.grid()
    plt.legend(loc=1)
    plt.show()

    x = 1
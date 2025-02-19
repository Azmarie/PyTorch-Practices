'''
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

NUM_EPOCH = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

######################################################
####### Do not modify the code above this line #######
######################################################

class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url = 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth'
        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]
        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)


if __name__ == '__main__':
    model = cifar_resnet20()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
 
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

    # Add testset and testloader
    # Adapted from Pytorch tutorials: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9)
    PATH = './cifar_net.pth'

    # import pdb; pdb.set_trace()
    import os.path
    if not os.path.exists(PATH):
        print("Do the training")
        for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
    else:
        # Load the .pth model for testing 
        model.load_state_dict(torch.load(PATH))

    # Get Test Set Accuracy for Task 2
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the testset: %d %%' % (
    #     100 * correct / total))

    def generate_html():
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<style>\n')
        f.write('table, th, td {\n')
        f.write('border: 1px solid black;\n')
        f.write('border-collapse: collapse;\n')
        f.write('}\n')
        f.write('</style>\n')
        f.write('</head>\n\n')
        f.write('<body>\n')
        f.write('<table style="width:100%">\n')
        f.write('<tr>\n')
        f.write('<th>Image</th>\n')
        f.write('<th>Plane</th>\n')
        f.write('<th>Car</th>\n')
        f.write('<th>Bird</th>\n')
        f.write('<th>Cat</th>\n')
        f.write('<th>Deer</th>\n')
        f.write('<th>Dog</th>\n')
        f.write('<th>Frog</th>\n')
        f.write('<th>Horse</th>\n')
        f.write('<th>Ship</th>\n')
        f.write('<th>Truck</th>\n')
        f.write('</tr>\n')

        with torch.no_grad():
            i = 0
            for data in testloader:
                images, labels = data
                outputs = model(images)

                # import pdb; pdb.set_trace()
                sm = nn.Softmax(dim=1)

                # Because the batch_size is 32
                for iter in range(16): 
                    outputs_softmax = sm(outputs)[iter].tolist()
                    formatted_list = ["%.3f"%item for item in outputs_softmax]
                    _, predicted = torch.max(outputs.data, 1)

                    f.write('  <tr>\n    <td align = "center">')
                    f.write('<img src="data/test/image' + str(32*i+iter) + '.png" alt="" border="1" /> ' + classes[labels[iter]] + ' </td>\n <td align = "center">')
                    f.write('</td>\n    <td align = "center">'.join(str(x) for x in formatted_list))
                    f.write('</td>\n  </tr>\n')

                    print(32*i+iter, classes[labels[iter]], classes[predicted[iter]]) # labels, predicted)
                i=i+1

        f.write('</table>\n')
        f.write('</body>\n')
        f.write('</html>\n')

    # Get Cifar classes from https://www.cs.toronto.edu/~kriz/cifar.html
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    f = open('result.html', 'w+')
    generate_html()
    f.close()
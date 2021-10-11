'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv

from utils import get_gaussian_filter

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
        ):
        super(BasicBlock, self).__init__()
        
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )



    def get_new_kernels(self, kernel_size, std):
        self.kernel1 = get_gaussian_filter(
                kernel_size=kernel_size,
                sigma=std,
                channels=self.planes,
        )
        self.kernel2 = get_gaussian_filter(
                kernel_size=kernel_size,
                sigma=std,
                channels=self.planes,
        )

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.std = args.std
        self.factor = args.std_factor
        self.epoch = args.epoch
        self.kernel_size = args.kernel_size
        self.precision_point = args.precision_point
        self.dataset= args.dataset

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, args.num_classes)

        self._initialize_weights()


        directory = "CSV_Data/"
        precision_point = str(self.precision_point)
        CSVFile = "ResNet18_"+self.dataset+"_Precision_Point_"+precision_point+".csv"
        self.cfile = directory + CSVFile
        current_working_dir = os.getcwd()
        filename = os.path.join(current_working_dir, self.cfile)
        print(filename)
        header = ['Data_Std', 'ResNet18_Layer_1_Std', 'ResNet18_Layer_5_Std', 'ResNet18_Layer_9_Std', 'ResNet18_Layer_13_Std',
                  'ResNet18_Layer_18_Std', 'Data_Mean', 'ResNet18_Layer_1_Mean', 'ResNet18_Layer_5_Mean', 'ResNet18_Layer_9_Mean',
                  'ResNet18_Layer_13_Mean', 'ResNet18_Layer_18_Mean','ResNet18_Std']
        with open(filename, mode='w') as write_obj:
            csvwriter = csv.writer(write_obj)
            csvwriter.writerow(header)

        # Data Writing for one iteration
        directory = "CSV_Data/Epoch/"
        CSVFile = "data" + ".csv"
        self.dataFile = directory + CSVFile
        print(self.dataFile)

    def setTrainingModelStatus(self, modelStatus):
        self.modelStatus = modelStatus

    def get_std(self, x, x1, x2, x3, x4,x5):
        with torch.no_grad():
            # Initial Image
            y = x.float()
            mean0 = torch.mean(y)
            sd0 = torch.std(y)
            # Layer One
            y1 = x1.float()
            mean1 = torch.mean(y1)
            sd1 = torch.std(y1)
            # Layer Two
            y2 = x2.float()
            mean2 = torch.mean(y2)
            sd2 = torch.std(y2)
            # Layer Three
            y3 = x3.float()
            mean3 = torch.mean(y3)
            sd3 = torch.std(y3)
            # Layer Four
            y4 = x4.float()
            mean4 = torch.mean(y4)
            sd4 = torch.std(y4)

            # Layer Five
            y5 = x5.float()
            mean5 = torch.mean(y5)
            sd5 = torch.std(y5)

        current_working_dir = os.getcwd()
        filename = os.path.join(current_working_dir, self.cfile)
        row = [sd0.item(), sd1.item(), sd2.item(), sd3.item(), sd4.item(),sd5.item(),
               mean0.item(), mean1.item(), mean2.item(),mean3.item(), mean4.item(), mean5.item(),
               self.std]
        with open(filename, mode='a') as write_obj:
            csvwriter = csv.writer(write_obj)
            csvwriter.writerow(row)

        # print(filename)
        dataFilename = os.path.join(current_working_dir, self.dataFile)
        row = [sd1.item(), sd2.item(), sd3.item(), sd4.item(),sd5.item()]
        with open(dataFilename, mode='a') as write_obj:
            csvwriter = csv.writer(write_obj)
            csvwriter.writerow(row)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = F.relu(self.bn1(x1))

        x2 = self.layer1(x1)

        x3 = self.layer2(x2)

        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        x5 = F.avg_pool2d(x5, 4)

        if self.modelStatus:
            self.get_std(x, x1, x2, x3, x4, x5)

        #print(x5.shape)
        x5 = x5.view(x5.size(0), -1)
        #print(x5.shape)
        x5 = self.linear(x5)
        return x5


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= self.factor
        self.kernel1 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=64,
        )

        for child in self.layer1.children():
            child.get_new_kernels(self.kernel_size, self.std)

        for child in self.layer2.children():
            child.get_new_kernels(self.kernel_size, self.std)

        for child in self.layer3.children():
            child.get_new_kernels(self.kernel_size, self.std)

        for child in self.layer4.children():
            child.get_new_kernels(self.kernel_size, self.std)



def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3,4,6,3], args)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3], args)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3], args)



def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


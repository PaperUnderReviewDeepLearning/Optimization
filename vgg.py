import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os
import csv
from utils import *


class VGG16_conv(torch.nn.Module):
    def __init__(self, n_classes, args):
        super(VGG16_conv, self).__init__()
        self.std = args.std
        self.factor = args.std_factor
        self.epoch = args.epoch
        self.kernel_size = args.kernel_size
        self.precision_point = args.precision_point
        self.dataset = args.dataset

        directory = "CSV_Data/"
        precision_point = str(self.precision_point)
        CSVFile = "VGG16_"+self.dataset+"_Precision_Point_"+precision_point+".csv"
        self.cfile = directory + CSVFile
        current_working_dir = os.getcwd()
        filename = os.path.join(current_working_dir, self.cfile)
        print(filename)
        header = ['Data_Std', 'VGG16_Layer_3_Std', 'VGG16_Layer_6_Std', 'VGG16_Layer_9_Std', 'VGG16_Layer_12_Std',
                  'VGG16_Layer_16_Std', 'Data_Mean', 'VGG16_Layer_3_Mean', 'VGG16_Layer_6_Mean', 'VGG16_Layer_9_Mean',
                  'VGG16_Layer_12_Mean', 'VGG16_Layer_16_Mean','VGG16_Std']
        with open(filename, mode='w') as write_obj:
            csvwriter = csv.writer(write_obj)
            csvwriter.writerow(header)

        # Data Writing for one iteration
        directory = "CSV_Data/Epoch/"
        CSVFile = "data" + ".csv"
        self.dataFile = directory + CSVFile
        print(self.dataFile)


        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
        )
        self.post1 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, padding=1),
        )
        self.post2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, padding=1),
        )
        self.post3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),
        )
        self.post4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv5 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),

        )
        self.post5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )

        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, n_classes)
        )
        
        self._initialize_weights()

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


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= 0.9

        self.kernel1 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=64
        )

        self.kernel2= get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=128
        )

        self.kernel3 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=256
        )

        self.kernel4 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )

        self.kernel5 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )
    def setTrainingModelStatus(self, modelStatus):
        self.modelStatus = modelStatus
    def get_std(self, x, x1, x2, x3, x4, x5):
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
      row = [sd0.item(), sd1.item(), sd2.item(), sd3.item(), sd4.item(), sd5.item(),
             mean0.item(), mean1.item(), mean2.item(), mean3.item(), mean4.item(), mean5.item(),
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

    def forward(self, x, return_intermediate=False):

        x1 = self.conv1(x)
        #x1 = self.kernel1(x1)
        x1 = self.post1(x1)

        x2 = self.conv2(x1)
        #x2 = self.kernel2(x2)
        x2 = self.post2(x2)

        x3 = self.conv3(x2)
        #x3 = self.kernel3(x3)
        x3 = self.post3(x3)

        x4 = self.conv4(x3)
        #x4 = self.kernel4(x4)
        x4 = self.post4(x4)

        x5 = self.conv5(x4)
        #x5 = self.kernel5(x5)

        if return_intermediate:
            output = x5.view(x5.size(0), -1)
            return output

        x5 = self.post5(x5)

        if self.modelStatus:
            self.get_std(x, x1, x2, x3, x4, x5)

        x5 = x5.view(x5.size(0), -1)
        output = self.classifier(x5)

        return output

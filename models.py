import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import math


def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

  mean = (kernel_size - 1) / 2.
  variance = sigma ** 2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                    torch.exp(
                      -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                      (2 * variance)
                    )

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).cuda()

  if kernel_size == 3:
    padding = 1
  else:
    padding = 0
  gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, groups=channels,
                              bias=False, padding=padding)

  gaussian_filter.weight.data = gaussian_kernel
  gaussian_filter.weight.requires_grad = False

  return gaussian_filter


class CNNNormal(nn.Module):
  def __init__(self, nc, num_classes,precision_point, dataset, std=1):
    super(CNNNormal, self).__init__()

    self.precision_point = precision_point
    self.dataset = dataset

    self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, padding=1)
    self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc = nn.Linear(256 * 2 * 2, 256 * 2 * 2) #for 32*32 Images
    self.classifier = nn.Linear(256 * 2 * 2, num_classes) #for 32*32 Images


    self.std = std
    directory = "CSV_Data/"
    precision_point = str(self.precision_point)
    CSVFile = "CNN_"+ self.dataset+"_Precision_Point_" + precision_point + ".csv"
    self.cfile = directory + CSVFile
    current_working_dir = os.getcwd()
    filename = os.path.join(current_working_dir, self.cfile)
    print(filename)
    header = ['Data_Std', 'CNN_Layer_1_Std', 'CNN_Layer_2_Std', 'CNN_Layer_3_Std', 'CNN_Layer_4_Std',
              'Data_Mean', 'CNN_Layer_1_Mean', 'CNN_Layer_2_Mean', 'CNN_Layer_3_Mean','CNN_Layer_4_Mean']
    with open(filename, mode='w') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(header)


    #Data Writing for one iteration
    directory = "CSV_Data/Epoch/"
    CSVFile = "data" + ".csv"
    self.dataFile = directory + CSVFile
    print(self.dataFile)

  def get_new_kernels(self, epoch_count):
    if epoch_count % 10 == 0:
      self.std *= 0.925
    self.kernel0 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=3)
    self.kernel1 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=32)
    self.kernel2 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=64)
    self.kernel3 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=128)
    self.kernel4 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=256)

  def get_std(self, x, x1, x2, x3, x4):

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





    current_working_dir = os.getcwd()
    filename = os.path.join(current_working_dir, self.cfile)
    row = [sd0.item(),sd1.item(), sd2.item(), sd3.item(), sd4.item(),
           mean0.item(), mean1.item(),mean2.item(), mean3.item(), mean4.item()]
    with open(filename, mode='a') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(row)



    #print(filename)
    dataFilename = os.path.join(current_working_dir, self.dataFile)
    row = [sd1.item(), sd2.item(), sd3.item(), sd4.item()]
    with open(dataFilename, mode='a') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(row)

  def setTrainingModelStatus(self,modelStatus):
    self.modelStatus = modelStatus

  def forward(self, x):

    #print("size of X as Images in forward function:", x.shape)
    #print(x.shape)
    x1 = self.conv1(x)
    x1 = F.relu(self.max1(x1))

    x2 = self.conv2(x1)
    x2 = F.relu(self.max2(x2))

    x3 = self.conv3(x2)
    x3 = F.relu(self.max3(x3))

    x4 = self.conv4(x3)
    x4 = F.relu(self.max4(x4))



    if self.modelStatus:
      self.get_std(x, x1, x2, x3, x4)

    x4 = x4.view(x4.size(0), -1)
    #print(x4.shape)

    x4 = F.relu(self.fc(x4))

    x4 = self.classifier(x4)

    return x4


class SimpleMLP(nn.Module):
  def __init__(self, num_classes, input_dim):
    super(SimpleMLP, self).__init__()

    self.fc1 = nn.Linear(input_dim, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, num_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class OneLayerMLP(nn.Module):
  def __init__(self, num_classes, input_dim):
    super(OneLayerMLP, self).__init__()
    self.fc1 = nn.Linear(input_dim, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    return x

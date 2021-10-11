import torch
import copy
import os
import torch.optim as optim
import torch.utils.data as data
import math
from sklearn.metrics import accuracy_score
from arguments import get_args
import csv
import pandas as pd
import sys
from termcolor import colored, cprint

#from models_gX5 import *
from data import get_data
from solver_base import BaseSolver

class CBSSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

        self.decay_epoch = 50 if self.args.alg == 'vgg' else 30
        self.stop_decay_epoch = self.decay_epoch * 3 + 1

    def meanCSV(self,alg):
        if alg == 'normal':
            column_name = ['layer1', 'layer2', 'layer3', 'layer4']

            directory = "CSV_Data/MeanData/"
            CSVFile = "data" + ".csv"
            self.cfile = directory + CSVFile
            current_working_dir = os.getcwd()
            filename = os.path.join(current_working_dir, self.cfile)
            print(filename)

            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)
            return filename
        elif alg == 'res':
            column_name = ['layer1', 'layer5', 'layer9', 'layer13','layer18']

            directory = "CSV_Data/MeanData/"
            CSVFile = "data" + ".csv"
            self.cfile = directory + CSVFile
            current_working_dir = os.getcwd()
            filename = os.path.join(current_working_dir, self.cfile)
            print(filename)

            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)
            return filename
        elif alg == 'vgg':
            column_name = ['layer3', 'layer6', 'layer9', 'layer12','layer16']

            directory = "CSV_Data/MeanData/"
            CSVFile = "data" + ".csv"
            self.cfile = directory + CSVFile
            current_working_dir = os.getcwd()
            filename = os.path.join(current_working_dir, self.cfile)
            print(filename)

            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)
            return filename


    def iterationCSV(self):

        directory = "CSV_Data/Epoch/"
        CSVFile = "data" + ".csv"
        self.cfile = directory + CSVFile
        current_working_dir = os.getcwd()
        filename = os.path.join(current_working_dir, self.cfile)

        if self.args.alg == 'normal':
            column_name = ['layer1', 'layer2', 'layer3', 'layer4']
            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)

        elif self.args.alg == 'res':
            column_name = ['layer1', 'layer5', 'layer9', 'layer13', 'layer18']
            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)
        elif self.args.alg == 'vgg':
            column_name = ['layer3', 'layer6', 'layer9', 'layer12', 'layer16']
            with open(filename, mode='w') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(column_name)

        return filename


    def solve(self):
        best_epoch, best_acc = 0, 0
        num_iter = 0


        meanFile = self.meanCSV(self.args.alg)


        for epoch_count in range(self.args.num_epochs):
            self.model.get_new_kernels(epoch_count)

            if self.args.epoch_limit == epoch_count:
                cprint("===Ending Training Model======","red","on_white")
                break

            iterationDataFile = self.iterationCSV() #Create CSV file for one epoch (all iteration)


            if self.cuda:
                self.model = self.model.cuda()

            if epoch_count is not 0 and epoch_count % self.decay_epoch == 0 \
                    and epoch_count < self.stop_decay_epoch:
                for param in self.optim.param_groups:
                    param['lr'] = param['lr'] / 10 

            for images, labels in self.train_data:

                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()


                #self.model.get_new_kernels(epoch_count,train_data=images) #change here
                self.model.setTrainingModelStatus(True)
                preds = self.model(images)
                loss = self.ce_loss(preds, labels)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step() 

                num_iter += 1

                if num_iter % 200 == 0:
                    print('iter num: {} \t loss: {:.2f}'.format(num_iter, loss.item()))

            if epoch_count % 1 == 0:
                self.model.setTrainingModelStatus(False)
                #accuracy = self.test()
                #if accuracy > best_acc:
                    #best_acc = accuracy
                    #best_epoch = epoch_count
                    #self.best_model = copy.deepcopy(self.model)

                print('epoch count: {}'.format(epoch_count))
                #print('best acc: {} \t best acc: {:.2f}'.format(best_epoch, best_acc))


            df = pd.read_csv(iterationDataFile) #Current Epoch mean CSV read


            if self.args.alg == 'normal':

                if epoch_count < 5:
                    pl1 = df['layer1'].mean().round(decimals=self.args.precision_point)
                    pl2 = df['layer2'].mean().round(decimals=self.args.precision_point)
                    pl3 = df['layer3'].mean().round(decimals=self.args.precision_point)
                    pl4 = df['layer4'].mean().round(decimals=self.args.precision_point)


                    row = [pl1,pl2,pl3,pl4]
                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)



                else:

                    l1 = df['layer1'].mean().round(decimals=self.args.precision_point)
                    l2 = df['layer2'].mean().round(decimals=self.args.precision_point)
                    l3 = df['layer3'].mean().round(decimals=self.args.precision_point)
                    l4 = df['layer4'].mean().round(decimals=self.args.precision_point)

                    row = [l1,l2,l3,l4]
                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)


                    if pl1 - l1 == 0:
                        cprint("layer 1 is Stable","red","on_white")
                        if pl2 - l2 == 0:
                            cprint("layer 2 is Stable","red","on_white")
                            if pl3 - l3 == 0:
                                cprint("layer 3 is stable","red","on_white")
                                if pl4-l4 == 0:
                                    cprint("layer 4 is stable","red","on_white")
                                    cprint("====Checking All Layers Previous Epochs===","red","on_white")
                                    dfMean = pd.read_csv(meanFile)
                                    dftail = dfMean.tail(5)
                                    first, flag = 0, False
                                    for i in dftail.index:
                                        first = i
                                        break
                                    for i in dftail.index:
                                        if i == first:
                                            continue
                                        elif dftail['layer1'][i]-dftail['layer1'][i - 1] == 0 and \
                                          dftail['layer2'][i] - dftail['layer2'][i - 1] == 0 and \
                                          dftail['layer3'][i] - dftail['layer3'][i - 1] == 0 and \
                                          dftail['layer4'][i] - dftail['layer4'][i - 1] == 0:
                                               flag = True
                                        else:
                                            flag = False
                                            break

                                    if flag:
                                        self.args.epoch_limit = epoch_count + 1

                    pl1,pl2,pl3,pl4 = l1,l2,l3,l4



            elif self.args.alg == 'res':
                if epoch_count < 5:
                    pl1 = df['layer1'].mean().round(decimals=self.args.precision_point)
                    pl2 = df['layer5'].mean().round(decimals=self.args.precision_point)
                    pl3 = df['layer9'].mean().round(decimals=self.args.precision_point)
                    pl4 = df['layer13'].mean().round(decimals=self.args.precision_point)
                    pl5 = df['layer18'].mean().round(decimals=self.args.precision_point)
                    row = [pl1, pl2, pl3, pl4,pl5]

                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)

                else:

                    l1 = df['layer1'].mean().round(decimals=self.args.precision_point)
                    l2 = df['layer5'].mean().round(decimals=self.args.precision_point)
                    l3 = df['layer9'].mean().round(decimals=self.args.precision_point)
                    l4 = df['layer13'].mean().round(decimals=self.args.precision_point)
                    l5 = df['layer18'].mean().round(decimals=self.args.precision_point)

                    row = [l1, l2, l3, l4,l5]
                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)

                    if pl1 - l1 == 0:
                        cprint("layer 1 is Stable", "red", "on_white")
                        if pl2 - l2 == 0:
                            cprint("layer 5 is Stable", "red", "on_white")
                            if pl3 - l3 == 0:
                                cprint("layer 9 is stable", "red", "on_white")
                                if pl4 - l4 == 0:
                                    cprint("layer 13 is stable", "red", "on_white")
                                    if pl5-l5 == 0:
                                        cprint("layer 18 is stable", "red", "on_white")
                                        cprint("====Checking All Layers Previous Epochs===", "red", "on_white")
                                        dfMean = pd.read_csv(meanFile)
                                        dftail = dfMean.tail(5)
                                        first, flag = 0, False
                                        for i in dftail.index:
                                            first = i
                                            break
                                        for i in dftail.index:
                                            if i == first:
                                                continue
                                            elif dftail['layer1'][i] - dftail['layer1'][i - 1] == 0 and \
                                              dftail['layer5'][i] - dftail['layer5'][i - 1] == 0 and \
                                              dftail['layer9'][i] - dftail['layer9'][i - 1] == 0 and \
                                              dftail['layer13'][i] - dftail['layer13'][i - 1] == 0 and \
                                              dftail['layer18'][i] - dftail['layer18'][i - 1] == 0:
                                                flag = True
                                            else:
                                                flag = False
                                                break

                                        if flag:
                                            self.args.epoch_limit = epoch_count + 1

                    pl1, pl2, pl3, pl4,pl5 = l1, l2, l3, l4,l5


            elif self.args.alg == 'vgg':
                if epoch_count < 5:
                    pl1 = df['layer3'].mean().round(decimals=self.args.precision_point)
                    pl2 = df['layer6'].mean().round(decimals=self.args.precision_point)
                    pl3 = df['layer9'].mean().round(decimals=self.args.precision_point)
                    pl4 = df['layer12'].mean().round(decimals=self.args.precision_point)
                    pl5 = df['layer16'].mean().round(decimals=self.args.precision_point)
                    row = [pl1, pl2, pl3, pl4, pl5]

                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)

                else:

                    l1 = df['layer3'].mean().round(decimals=self.args.precision_point)
                    l2 = df['layer6'].mean().round(decimals=self.args.precision_point)
                    l3 = df['layer9'].mean().round(decimals=self.args.precision_point)
                    l4 = df['layer12'].mean().round(decimals=self.args.precision_point)
                    l5 = df['layer16'].mean().round(decimals=self.args.precision_point)

                    row = [l1, l2, l3, l4, l5]
                    with open(meanFile, mode='a') as write_obj:
                        csvwriter = csv.writer(write_obj)
                        csvwriter.writerow(row)

                    if pl1 - l1 == 0:
                        cprint("layer 3 is Stable", "red", "on_white")
                        if pl2 - l2 == 0:
                            cprint("layer 6 is Stable", "red", "on_white")
                            if pl3 - l3 == 0:
                                cprint("layer 9 is stable", "red", "on_white")
                                if pl4 - l4 == 0:
                                    cprint("layer 12 is stable", "red", "on_white")
                                    if pl5 - l5 == 0:
                                        cprint("layer 16 is stable", "red", "on_white")
                                        cprint("====Checking All Layers Previous Epochs===", "red", "on_white")
                                        dfMean = pd.read_csv(meanFile)
                                        dftail = dfMean.tail(5)
                                        first, flag = 0, False
                                        for i in dftail.index:
                                            first = i
                                            break
                                        for i in dftail.index:
                                            if i == first:
                                                continue
                                            elif dftail['layer3'][i] - dftail['layer3'][i - 1] == 0 and \
                                              dftail['layer6'][i] - dftail['layer6'][i - 1] == 0 and \
                                              dftail['layer9'][i] - dftail['layer9'][i - 1] == 0 and \
                                              dftail['layer12'][i] - dftail['layer12'][i - 1] == 0 and \
                                              dftail['layer16'][i] - dftail['layer16'][i - 1] == 0:
                                                flag = True
                                            else:
                                                flag = False
                                                break

                                        if flag:
                                            self.args.epoch_limit = epoch_count + 1

                    pl1, pl2, pl3, pl4, pl5 = l1, l2, l3, l4, l5

        print("=====Testing Model====")
        self.model.setTrainingModelStatus(False)
        self.model.eval()
        total, correct,accuracyE = 0, 0, 0
        for images, labels in self.test_data:
            if self.cuda:
                images = images.cuda()

            with torch.no_grad():
                preds = self.model(images)
                preds = torch.argmax(preds, dim=1).cpu().numpy()

                correct += accuracy_score(labels, preds, normalize=False)
                total += images.size(0)

        accuracyValidation=correct / total * 100
        #self.model.train()
        print('Accuracy: ', accuracyValidation)
        '''
            current_working_dir = os.getcwd()
            filename = os.path.join(current_working_dir, cfile)
            row = [epoch_count,accuracy,accuracyValidation]
            with open(filename, mode='a') as write_obj:
                csvwriter = csv.writer(write_obj)
                csvwriter.writerow(row)
        '''

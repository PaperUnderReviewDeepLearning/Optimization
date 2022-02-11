# When do Convolutional Neural Networks Stop Learning?
#Paper Under Review KDD 2022

The official PyTorch implementation of When do Convolutional Neural Networks Stop Learning? (KDD 2022 #Paper Under Review#).

Required Packages:

python 3.6.3

numpy 1.19.2

pandas 1.1.5

pytorch 1.3.1

scipy 1.5.2

seaborn 0.11.1

termcolor 1.1.0

torchvision

scikit-learn

***USAGE:***
Simply use the code by running:

`python3 main.py --dataset <DATASET> --alg <MODEL> --data <PATH_TO_DATA>`

For example, to train a ResNet on CIFAR10 and the data is saved in `./data/`, we can run:

`python3 main.py --dataset cifar10 --alg res --data ./data/`


To use CBS models please modify the file name by following:

CNN+CBS:        models_CBS.py ->  models.py

ResNet18+CBS:   resnet_CBS.py ->  resent.py

VGG16+CBS:      Vgg_CBS.py    ->  Vgg.py




# When do Convolutional Neural Networks Stop Learning?
#Paper Under Review

The official PyTorch implementation of When do Convolutional Neural Networks Stop Learning? (ICLR 2022#Paper Under Review#).



***USAGE:***
Simply use the code by running:

`python3 main.py --dataset <DATASET> --alg <MODEL> --data <PATH_TO_DATA>`

For example, to train a ResNet on CIFAR10 and the data is saved in `./data/`, we can run:

`python3 main.py --dataset cifar10 --alg res --data ./data/`


For new expeiments it is important to tune the following hyperparameters:

`--std --std_factor --epoch`



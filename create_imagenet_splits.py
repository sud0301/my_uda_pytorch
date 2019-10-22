'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import os
import argparse
import pickle
import numpy as np
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from utils import progress_bar
from augment.cutout import Cutout

#from wrn import wrn
#from autoaugment_extra_only_color import CIFAR10Policy
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy

from utils import Logger, AverageMeter, mkdir_p, savefig, progress_bar


#Models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


DATASET = 'ImageNet' # 'CIFAR10'
SEED = 3
SPLIT_ID = None

parse = argparse.ArgumentParser(description='PyTorch SSL CIFAR10 UDA Training')
parse.add_argument('--dataset', type=str, default=DATASET, help='dataset')
parse.add_argument('-d', '--imagenet-data', default='/misc/lmbraid19/mittal/my_repos/cloned_repos/pytorch-classification/data/', type=str)
parse.add_argument('--num-classes', default=1000, type=int, help='number of classes')

parse.add_argument('--percent-labeled', default=10, type=float, help='percentage of labeled_samples')

parse.add_argument('--start-iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts)')

parse.add_argument('--split-id', type=str, default=SPLIT_ID, help='restore partial id list')
parse.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parse.add_argument('--seed', default=SEED, type=int, help='seed index')

# Architectures 

parse.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parse.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parse.parse_args()
state = {k: v for k, v in args._get_kwargs()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
np.random.seed(args.seed)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# SEED
if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

def main():
    global best_acc
    start_iter = args.start_iter  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    traindir = os.path.join(args.imagenet_data, 'train')
    trainset = torchvision.datasets.ImageFolder(traindir, transform=transform_test)

    
    train_dataset_size = len(trainset)

    #print ('train dataset size: ', train_dataset_size)
    #train_dataset_size = 10000

    train_ids = np.arange(train_dataset_size)
    np.random.shuffle(train_ids)
    #train_ids = train_ids[:10000]
    #pickle.dump(train_ids, open(os.path.join(args.checkpoint, 'train_id_' + str(args.seed) + '.pkl'), 'wb'))

    mask = np.zeros(train_ids.shape[0], dtype=np.bool)
    labels = np.array([trainset[i][1] for i in train_ids], dtype=np.int64)
    num_labeled = int(args.percent_labeled*len(train_ids)/100.0)
    print ('Num labeled: ', num_labeled)
    for i in range(args.num_classes):
        mask[np.where(labels == i)[0][: int(num_labeled / args.num_classes)]] = True
    labeled_indices, unlabeled_indices = train_ids[mask], train_ids[~ mask]
        
    pickle.dump(labeled_indices, open(os.path.join(args.checkpoint, 'labeled_idxs_' + str(args.percent_labeled) + '_seed_' + str(args.seed) + '_round_0.pkl'), 'wb'))
    pickle.dump(unlabeled_indices, open(os.path.join(args.checkpoint, 'unlabeled_idxs_' + str(args.percent_labeled) + '_seed_' + str(args.seed) + '_round_0.pkl'), 'wb'))

    print ('Labeled indices: ', len(labeled_indices), ' Unlabeled indices: ', len(unlabeled_indices))

    for rd in range(1, 5):
        idxs_unlabeled = np.arange(train_ids.shape[0])[unlabeled_indices]
        print ('len of unlabeled indices: ', len(idxs_unlabeled))
        labeled_indices =  np.random.choice(idxs_unlabeled, num_labeled, replace=False)
        unlabeled_indices =  np.setdiff1d(idxs_unlabeled, labeled_indices)
        pickle.dump(labeled_indices, open(os.path.join(args.checkpoint, 'labeled_idxs_' + str(args.percent_labeled) + '_seed_' + str(args.seed) + '_round_' + str(rd) + '.pkl'), 'wb'))
        pickle.dump(unlabeled_indices, open(os.path.join(args.checkpoint, 'unlabeled_idxs_' + str(args.percent_labeled) + '_seed_' + str(args.seed) + '_round_' + str(rd) + '.pkl'), 'wb'))
        print ('Labeled indices: ', len(labeled_indices), ' Unlabeled indices: ', len(unlabeled_indices))
        

if __name__ == '__main__':
    main() 

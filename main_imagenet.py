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
#from utils import progress_bar
from augment.cutout import Cutout

#from wrn import wrn
#from autoaugment_extra_only_color import CIFAR10Policy
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy

from utils import Logger, AverageMeter, mkdir_p, savefig#, progress_bar


#Models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

torch.backends.cudnn.enabled = True

DATASET = 'ImageNet' # 'CIFAR10'
SEED = 1
SPLIT_ID = None

parse = argparse.ArgumentParser(description='PyTorch SSL CIFAR10 UDA Training')
parse.add_argument('--dataset', type=str, default=DATASET, help='dataset')
#parse.add_argument('-d', '--imagenet-data', default='/misc/lmbraid19/mittal/my_repos/cloned_repos/pytorch-classification/data/', type=str)
parse.add_argument('-d', '--imagenet-data-train', default='/misc/lmbssd/marrakch/ILSVRC2015/Data/CLS-LOC/', type=str)
parse.add_argument('--imagenet-data-val', default='/misc/lmbraid19/mittal/my_repos/cloned_repos/pytorch-classification/data/', type=str)
parse.add_argument('--num-classes', default=1000, type=int, help='number of classes')

parse.add_argument('--lr', default=0.3, type=float, help='learning rate')
parse.add_argument('--softmax-temp', default=-1, type=float, help='softmax temperature controlling')
parse.add_argument('--confidence-mask', default=-1, type=float, help='Confidence value for masking')

parse.add_argument('--num-labeled', default=10000, type=int, help='number of labeled_samples')
parse.add_argument('--percent-labeled', default=20, type=float, help='percentage of labeled_samples')

parse.add_argument('--start-iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts)')

parse.add_argument('--batch-size-lab', default=64, type=int, help='training batch size')
parse.add_argument('--batch-size-unlab', default=640, type=int, help='training batch size')
parse.add_argument('--num-steps', default=100000, type=int, help='number of iterations')
parse.add_argument('--num-epochs', default=40, type=int, help='number of iterations')
parse.add_argument('--lr-warm-up', action='store_true', help='increase lr slowly')
parse.add_argument('--warm-up-steps', default=20000, type=int, help='number of iterations for warmup')

parse.add_argument('--split-id', type=str, default=SPLIT_ID, help='restore partial id list')
parse.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parse.add_argument('--new-splits', action='store_true', help='resume same splits')
parse.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parse.add_argument('--verbose', action='store_true', help='show progress bar')
parse.add_argument('--seed', default=SEED, type=int, help='seed index')

# Architectures 
parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parse.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parse.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# Supervised or Semi-supervised
parse.add_argument('--lab-only', action='store_true', help='if using only labeled samples')

# Augmenatations
parse.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parse.add_argument('--n-holes', default=1, type=float, help='number of holes for cutout')
parse.add_argument('--cutout-size', default=16, type=float, help='size of the cutout window')
parse.add_argument('--autoaugment', action='store_true', help='use autoaugment augmentation')
parse.add_argument('--loss-lambda', default=1.0, type=float, help='weightage of the unlabeled loss')

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

    # Data
    print('==> Preparing data..')
    if args.dataset == 'CIFAR10':
        transform_ori = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_aug = transforms.Compose([
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=args.n_holes, length=args.cutout_size),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    elif args.dataset == 'ImageNet':
        transform_ori = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_aug = transforms.Compose([
            ImageNetPolicy(),
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=56),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_train = TransformTwice(transform_ori, transform_aug)
    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        labelset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        labelset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'ImageNet':
        traindir = os.path.join(args.imagenet_data_train, 'train')
        valdir = os.path.join(args.imagenet_data_val, 'val')
        trainset = torchvision.datasets.ImageFolder(traindir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(valdir, transform=transform_test)    

    train_dataset_size = len(trainset)
    test_dataset_size = len(testset)

    print ('train dataset size: ', train_dataset_size, 'test dataset size: ', test_dataset_size)

    train_ids = np.arange(train_dataset_size)
    test_ids = np.arange(test_dataset_size)
    np.random.shuffle(train_ids)
    #pickle.dump(train_ids, open(os.path.join(args.checkpoint, 'train_id_' + str(args.seed) + '.pkl'), 'wb'))

    if args.lab_only:
        num_labeled = int(args.percent_labeled*len(train_ids)/100.0)
        print ('Num labeled: ', num_labeled)
        train_ids = train_ids[:num_labeled]
    
        train_sampler_lab = data.sampler.SubsetRandomSampler(train_ids)
        train_sampler_unlab = data.sampler.SubsetRandomSampler(train_ids)

        trainloader_lab = data.DataLoader(trainset, batch_size=args.batch_size_lab, sampler=train_sampler_lab, num_workers=12, drop_last=True, pin_memory=True)
        trainloader_unlab = data.DataLoader(trainset, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=1)
        
        #args.num_steps = int((num_labeled/args.batch_size_lab)*args.num_epochs)
        #print ('NUM STEPS: ', args.num_steps)

    else:
        '''
        if args.new_splits:
            mask = np.zeros(train_ids.shape[0], dtype=np.bool)
            labels = np.array([trainset[i][1] for i in train_ids], dtype=np.int64)
            num_labeled = int(args.percent_labeled*len(train_ids)/100.0)
            print ('Num labeled: ', num_labeled)
            for i in range(args.num_classes):
                mask[np.where(labels == i)[0][: int(num_labeled / args.num_classes)]] = True
            labeled_indices, unlabeled_indices = train_ids[mask], train_ids[~ mask]

            pickle.dump(labeled_indices, open(os.path.join(args.checkpoint, 'labeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'wb'))
            pickle.dump(unlabeled_indices, open(os.path.join(args.checkpoint, 'unlabeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'wb'))
        else:
            labeled_indices = pickle.load(open(os.path.join(args.checkpoint, 'labeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'rb'))
            unlabeled_indices = pickle.load(open(os.path.join(args.checkpoint, 'unlabeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'rb'))
        '''
        num_labeled = int(args.percent_labeled*len(train_ids)/100.0)
        print ('Num labeled: ', num_labeled)
        #train_ids = train_ids[:num_labeled]
        labeled_indices = train_ids[:num_labeled]
        unlabeled_indices = train_ids[num_labeled:]

        print ('Labeled indices: ', len(labeled_indices), ' Unlabeled indices: ', len(unlabeled_indices))
        
        train_sampler_lab = data.sampler.SubsetRandomSampler(labeled_indices)
        train_sampler_unlab = data.sampler.SubsetRandomSampler(unlabeled_indices)

        trainloader_lab = data.DataLoader(trainset, batch_size=args.batch_size_lab, sampler=train_sampler_lab, num_workers=6, drop_last=True)
        trainloader_unlab = data.DataLoader(trainset, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=6)
        #args.num_steps = int((num_labeled/args.batch_size_lab)*args.num_epochs)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = models.__dict__[args.arch]()
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=0.001)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=0.001)
    
    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_iter = checkpoint['iter']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Valid Acc.'])

    '''
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    '''
    train(trainloader_lab, trainloader_unlab, testloader, net, optimizer, criterion, start_iter, logger) 

def _kl_divergence_with_logits(p_logits, q_logits):
    p = torch.nn.functional.softmax(p_logits, dim=1)
    log_p = torch.nn.functional.log_softmax(p_logits, dim=1)
    log_q = torch.nn.functional.log_softmax(q_logits, dim=1)

    kl = torch.sum(p * (log_p - log_q), dim=1)
    return kl

def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Training
def train(trainloader_lab, trainloader_unlab, testloader, net, optimizer, criterion, start_iter, logger):
    
    train_loss = AverageMeter()
    train_loss_lab = AverageMeter()
    train_loss_unlab = AverageMeter()

    trainloader_lab_iter = iter(trainloader_lab)
    trainloader_unlab_iter = iter(trainloader_unlab)
  
    for i_iter in range(start_iter, args.num_steps):
        net.train()
        optimizer.zero_grad()
    
        if args.lr_warm_up:
            if i_iter < args.warm_up_steps:
                warmup_lr = i_iter/args.warm_up_steps* args.lr
                optimizer = set_optimizer_lr(optimizer, warmup_lr)
  
        if i_iter==int(args.num_steps/3):
            args.lr = args.lr/10
            optimizer = set_optimizer_lr(optimizer, args.lr)
        
        if i_iter==int(2*args.num_steps/3):
            args.lr = args.lr/10
            optimizer = set_optimizer_lr(optimizer, args.lr)

        if i_iter==int(8*args.num_steps/9):
            args.lr = args.lr/10
            optimizer = set_optimizer_lr(optimizer, args.lr)
            
        
        try:
            batch_lab = next(trainloader_lab_iter)
        except:
            trainloader_lab_iter = iter(trainloader_lab)
            batch_lab = next(trainloader_lab_iter) 
        
        (inputs_lab, _), targets_lab = batch_lab
        inputs_lab, targets_lab = inputs_lab.to(device), targets_lab.to(device)
       
        
        outputs_lab = net(inputs_lab)
        loss_lab = criterion(outputs_lab, targets_lab)
        #inputs, targets = inputs.to(device), targets.to(device)
        train_loss_lab.update(loss_lab.item())

        if args.lab_only:  
            loss_unlab = 0.0      
            loss = loss_lab 
        else:
            try:
                batch_unlab = next(trainloader_unlab_iter)
            except:
                trainloader_unlab_iter = iter(trainloader_unlab)
                batch_unlab = next(trainloader_unlab_iter) 
            
            (inputs_unlab, inputs_unlab_aug), _ = batch_unlab
            inputs_unlab, inputs_unlab_aug = inputs_unlab.cuda(), inputs_unlab_aug.cuda()
           
            outputs_unlab = net(inputs_unlab)
            outputs_unlab_aug = net(inputs_unlab_aug)

            if args.softmax_temp != -1:
               
                loss_unlab = _kl_divergence_with_logits(
                        p_logits=(outputs_unlab/args.softmax_temp).detach(),
                        q_logits=outputs_unlab_aug)
         
                if args.confidence_mask != -1:
                    unlab_prob = torch.nn.functional.softmax(outputs_unlab, dim=1)
                    largest_prob, _ = unlab_prob.max(1)
                    mask = (largest_prob>args.confidence_mask).float().detach()
                    loss_unlab = loss_unlab*mask 
               
                loss_unlab = torch.mean(loss_unlab)

            else:
                loss_unlab = torch.nn.functional.kl_div( 
                         torch.nn.functional.log_softmax(outputs_unlab_aug, dim=1), 
                         torch.nn.functional.softmax(outputs_unlab, dim=1).detach(), reduction='batchmean') 
       
            train_loss_unlab.update(loss_unlab.item())
        loss = loss_lab + args.loss_lambda*loss_unlab
        train_loss.update(loss.item())

        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        #train_loss += loss.item()
        #train_loss_lab += loss_lab.item()
        #if args.lab_only:
        #    train_loss_unlab = 0.0
        #else:
        #    train_loss_unlab += loss_unlab.item()
        
        if args.verbose:
            progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f | Loss_unlab: %.6f'
                % (train_loss.avg, train_loss_lab.avg, train_loss_unlab.avg))
        else: 
            if i_iter%1000==0:
                print (i_iter, ' Train loss: ', train_loss.avg, ' Loss lab: ', train_loss_lab.avg, ' Loss unlab: ', train_loss_unlab.avg)

        if i_iter%5000==0 and i_iter>0:
            test_loss, test_acc = test(net, testloader, criterion, optimizer, i_iter)
            logger.append([state['lr'], train_loss.avg, test_loss, test_acc])

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
       
def test(net, testloader, criterion, optimizer, i_iter):
    global best_acc

    test_loss = AverageMeter()

    net.eval()
    correct = 0
    total = 0
    U_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss.update(loss.item(), inputs.size(0))
    
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss.avg, 100.*correct/total, correct, total))
            else:
                print('Test loss: ', test_loss.avg, ' Accuracy: ', 100.*correct/total)
     
    # Save checkpoint.
    test_acc = 100.*correct/total

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
            'iter': i_iter,
            'state_dict': net.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            #'scheduler' : scheduler.state_dict(),
        }, is_best, checkpoint=args.checkpoint)
    
    return (test_loss.avg, test_acc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
    
#for cycle in range(args.num_cycles):
#    train(cycle, trainloader_lab, trainloader_unlab, scheduler, optimizer)

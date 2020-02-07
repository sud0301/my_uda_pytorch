'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from copy import deepcopy
import os
import argparse
import pickle
import numpy as np

#import dataloader
#from dataloader import DataLoader

#from models.senet import *
from utils import progress_bar, AverageMeter
from augment.cutout import Cutout

#from resnet import resnet18
from wrn import wrn
from densenet import *
#from autoaugment_extra_only_color import CIFAR10Policy
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy
from augment import randaugment
from vat import VATLoss
from resnet_pytorch import *

DATASET = 'CIFAR10'
SEED = 3
SPLIT_ID = None


parse = argparse.ArgumentParser(description='PyTorch SSL CIFAR10 UDA Training')
parse.add_argument('--dataset', type=str, default=DATASET, help='dataset')
parse.add_argument('-d', '--imagenet-data-train', default='/misc/lmbssd/marrakch/ILSVRC2015/Data/CLS-LOC/', type=str)
parse.add_argument('--imagenet-data-val', default='/misc/lmbraid19/mittal/my_repos/cloned_repos/pytorch-classification/data/', type=str)
#parse.add_argument('--arch', type=str, default=ARCH, help='dataset')
parse.add_argument('--num-classes', default=10, type=int, help='number of classes')

parse.add_argument('--lr', default=0.03, type=float, help='learning rate')
parse.add_argument('--wdecay', default=5e-4, type=float, help='learning weight decay')
parse.add_argument('--softmax-temp', default=-1, type=float, help='softmax temperature controlling')
parse.add_argument('--confidence-mask', default=-1, type=float, help='Confidence value for masking')

parse.add_argument('--num-labeled', default=1000, type=int, help='number of labeled_samples')
parse.add_argument('--percent-labeled', default=20, type=float, help='percentage of labeled_samples')

parse.add_argument('--batch-size-lab', default=32, type=int, help='training batch size')
parse.add_argument('--batch-size-unlab', default=320, type=int, help='training batch size')
parse.add_argument('--num-steps', default=100000, type=int, help='number of iterations')
parse.add_argument('--lr-warm-up', action='store_true', help='increase lr slowly')
parse.add_argument('--warm-up-steps', default=20000, type=int, help='number of iterations for warmup')
parse.add_argument('--num-cycles', default=1, type=int, help='number of sgdr cycles')

parse.add_argument('--split-id', type=str, default=SPLIT_ID, help='restore partial id list')
parse.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parse.add_argument('--verbose', action='store_true', help='show progress bar')
parse.add_argument('--seed', default=SEED, type=int, help='seed index')

# Supervised or Semi-supervised
parse.add_argument('--lab-only', action='store_true', help='if using only labeled samples')

# Augmenatations
parse.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parse.add_argument('--n-holes', default=1, type=float, help='number of holes for cutout')
parse.add_argument('--cutout-size', default=16, type=float, help='size of the cutout window')
parse.add_argument('--autoaugment', action='store_true', help='use autoaugment augmentation')

# Loss weighting parameters
parse.add_argument('--w-lab', default=0.3, type=float, help='weightage of sup loss')
parse.add_argument('--cons', action='store_true', help='use consistency loss')
parse.add_argument('--w-cons', default=0.3, type=float, help='weightage of consistency loss')
parse.add_argument('--rot', action='store_true', help='use rotation loss')
parse.add_argument('--w-rot', default=0.7, type=float, help='weightage of rot loss')
parse.add_argument('--vat', action='store_true', help='use rotation loss')
parse.add_argument('--w-vat', default=0.3, type=float, help='weightage of rot loss')
parse.add_argument('--entmin', action='store_true', help='use entropy minimization loss')
parse.add_argument('--w-entmin', default=0.3, type=float, help='weightage of entmin loss')
parse.add_argument('--pl', action='store_true', help='use pseudo labelling from fixmatch')
parse.add_argument('--w-pl', default=0.3, type=float, help='weightage of pseudo labelling')
parse.add_argument('--pl-conf-mask', default=0.95, type=float, help='Confidence value for pseudo labeling masking')

parse.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# EMA model
parse.add_argument('--use-ema', action='store_true', help='EMA model')
parse.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay parameter')

args = parse.parse_args()

#CHECKPOINT_DIR = './results/dataset_' + str(args.dataset)  + '_labels_' + str(args.num_labeled) + '_batch_lab_' +  str(args.batch_size_lab) + '_batch_unlab_' + str(args.batch_size_unlab)  + '_steps_' + str(args.num_steps) +'_warmup_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '_SEED_' + str(args.seed)
CHECKPOINT_DIR = './results/dataset_' + str(args.dataset) + '_labels_' + str(args.num_labeled) + '_batch_lab_' +  str(args.batch_size_lab) + '_batch_unlab_' + str(args.batch_size_unlab)  + '_steps_' + str(args.num_steps) +'_warmup_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '_SEED_' + str(args.seed) + '_cons_' + str(int(args.cons)) + '_rot_' + str(int(args.rot)) + '_pl_' + str(int(args.pl))

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
np.random.seed(args.seed)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def rotate_90(img, n):
    return np.rot90(img, n, (1, 2))

def collate(batch):
    rot_imgs = []
    rot_labels = []
    imgs = []
    aug_imgs = []
    for ((x, x_aug),_) in batch:
        # TODO: we can try randomizing the rotation here. 
        for n in [0, 1, 2, 3]:
            rot_imgs.append(torch.FloatTensor(rotate_90(x_aug.numpy(), n).copy()))
            rot_labels.append(torch.tensor(n))
        imgs.append(x)
        aug_imgs.append(x_aug)

    return [torch.stack(imgs), torch.stack(aug_imgs), torch.stack(rot_imgs), torch.stack(rot_labels)]

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.wd = args.lr * args.wdecay
        self.ema.to(device)
        self.ema_has_module = hasattr(self.ema, 'module')
        #if resume:
        #    self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'ema_state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['ema_state_dict'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                model_v = model_v.to(device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # weight decay
                if 'bn' not in k:
                    msd[k] = msd[k] * (1. - self.wd)

# Data
print('==> Preparing data..')
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    transform_ori = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_aug = transforms.Compose([
        transforms.ToTensor(),
        Cutout(n_holes=args.n_holes, length=args.cutout_size),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

elif args.dataset == 'ImageNet':
    transform_ori = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
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
    #labelset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    #labelset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'ImageNet':
    traindir = os.path.join(args.imagenet_data_train, 'train')
    valdir = os.path.join(args.imagenet_data_val, 'val')
    trainset = torchvision.datasets.ImageFolder(traindir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(valdir, transform=transform_test)
#trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

train_dataset_size = len(trainset)
test_dataset_size = len(testset)

#if args.partial_id is not None:
#train_ids = pickle.load(open(args.split_id, 'rb'))
#print('loading train ids from {}'.format(args.split_id))
#else:
#    train_ids = np.arange(train_dataset_size)
#    np.random.shuffle(train_ids)

#pickle.dump(train_ids, open('train_id.pkl', 'wb'))

if args.split_id is not None:
    train_ids = pickle.load(open(args.split_id, 'rb'))
    print('loading train ids from {}'.format(args.split_id))
else:
    train_ids = np.arange(train_dataset_size)
    test_ids = np.arange(test_dataset_size)
    np.random.shuffle(train_ids)
    pickle.dump(train_ids, open(os.path.join(CHECKPOINT_DIR, 'train_id_' + str(args.seed) + '.pkl'), 'wb'))

mask = np.zeros(train_ids.shape[0], dtype=np.bool)
labels = np.array([trainset[i][1] for i in train_ids], dtype=np.int64)
'''
for i in range(args.num_classes):
    mask[np.where(labels == i)[0][: int(args.num_labeled / args.num_classes)]] = True
labeled_indices, unlabeled_indices = train_ids[mask], train_ids[~ mask]
#labeled_indices, unlabeled_indices = train_ids[mask], train_ids
'''
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    mask_ = np.zeros(train_ids.shape[0], dtype=np.bool)
    #mask_test = np.zeros(test_ids.shape[0], dtype=np.bool)
    #labels_test = np.array([testset[i][1] for i in test_ids], dtype=np.int64)
    for i in range(args.num_classes):
        mask[np.where(labels == i)[0][: int(args.num_labeled / args.num_classes)]] = True
        mask_[np.where(labels == i)[0][int(args.num_labeled / args.num_classes): ]] = True
        #mask_test[np.where(labels_test == i)[0]] = True
    labeled_indices, unlabeled_indices = train_ids[mask], train_ids[mask_]
    #test_indices = test_ids[mask_test]
elif args.dataset == 'ImageNet':
    labeled_indices = pickle.load(open(os.path.join('./splits/labeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'rb'))
    unlabeled_indices = pickle.load(open(os.path.join('./splits/unlabeled_idxs_' + str(args.percent_labeled) + '_' + str(args.seed) + '.pkl'), 'rb'))

train_sampler_lab = data.sampler.SubsetRandomSampler(labeled_indices)
train_sampler_unlab = data.sampler.SubsetRandomSampler(unlabeled_indices)
#test_sampler = data.sampler.SubsetRandomSampler(test_indices)

trainloader_lab = data.DataLoader(trainset, batch_size=args.batch_size_lab, sampler=train_sampler_lab, num_workers=4, drop_last=True)
if args.rot:
    trainloader_unlab = data.DataLoader(trainset, collate_fn=collate, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=8, pin_memory=True)
else:
    trainloader_unlab = data.DataLoader(trainset, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=8, pin_memory=True)

#trainloader_val = data.DataLoader(labelset, batch_size=100, sampler=train_sampler_lab, num_workers=8, drop_last=False)

testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    net = wrn(num_classes=args.num_classes).cuda()
elif args.dataset == 'ImageNet':
    net = resnet50().cuda()
#net = densenet_cifar().cuda()
#net = models.resnet18().cuda()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    if args.use_ema:
        ema_net = ModelEMA(args, net, args.ema_decay)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=0.0001)

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Training
def train(cycle, trainloader_lab, trainloader_unlab, scheduler, optimizer):

    print ('count parameters: ', count_parameters(net))
    print('\nCycle: %d' % cycle)

    train_loss = AverageMeter()
    train_loss_lab = AverageMeter()
    train_loss_cons = AverageMeter()
    train_loss_rot = AverageMeter()
    train_loss_entmin = AverageMeter()
    train_loss_vat = AverageMeter()
    train_loss_pl = AverageMeter()
    correct = 0
    total = 0

    trainloader_lab_iter = iter(trainloader_lab)
    trainloader_unlab_iter = iter(trainloader_unlab)
  
    for i_iter in range(args.num_steps):
        net.train()
        optimizer.zero_grad()
    
        if args.lr_warm_up:
            if i_iter < args.warm_up_steps:
                warmup_lr = i_iter/args.warm_up_steps* args.lr
                optimizer = set_optimizer_lr(optimizer, warmup_lr)

        if args.dataset == 'ImageNet':
            if i_iter==int(3*args.num_steps/10):
                args.lr = args.lr/10
                optimizer = set_optimizer_lr(optimizer, args.lr)

            if i_iter==int(1*args.num_steps/2):
                args.lr = args.lr/10
                optimizer = set_optimizer_lr(optimizer, args.lr)

            if i_iter==int(3*args.num_steps/4):
                args.lr = args.lr/10
                optimizer = set_optimizer_lr(optimizer, args.lr)

            if i_iter==int(9*args.num_steps/10):
                args.lr = args.lr/10
                optimizer = set_optimizer_lr(optimizer, args.lr)

        if i_iter%1000==0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        
        try:
            batch_lab = next(trainloader_lab_iter)
        except:
            trainloader_lab_iter = iter(trainloader_lab)
            batch_lab = next(trainloader_lab_iter) 
        
        (inputs_lab, _), targets_lab = batch_lab
        inputs_lab, targets_lab = inputs_lab.to(device), targets_lab.to(device)
       
        outputs_lab, _ = net(inputs_lab)
        loss_lab = criterion(outputs_lab, targets_lab)

        loss = args.w_lab*loss_lab 
        
        try:
            batch_unlab = next(trainloader_unlab_iter)
        except:
            trainloader_unlab_iter = iter(trainloader_unlab)
            batch_unlab = next(trainloader_unlab_iter) 

        if args.rot:
            [inputs_unlab, inputs_unlab_aug, inputs_rot, targets_rot] = batch_unlab           
            inputs_rot, targets_rot = inputs_rot.cuda(), targets_rot.cuda()
        else:
            (inputs_unlab, inputs_unlab_aug), _ = batch_unlab

        if args.rot:
            _, outputs_rot = net(inputs_rot)
            loss_rot = criterion(outputs_rot, targets_rot)
            loss +=  args.w_rot*loss_rot

        if args.cons:
            inputs_unlab, inputs_unlab_aug = inputs_unlab.cuda(), inputs_unlab_aug.cuda()
           
            outputs_unlab, _ = net(inputs_unlab)
            outputs_unlab_aug, _ = net(inputs_unlab_aug)

            if args.softmax_temp != -1:
               
                loss_cons = _kl_divergence_with_logits(
                        p_logits=(outputs_unlab/args.softmax_temp).detach(),
                        q_logits=outputs_unlab_aug)
         
                if args.confidence_mask != -1:
                    unlab_prob = torch.nn.functional.softmax(outputs_unlab, dim=1)
                    largest_prob, _ = unlab_prob.max(1)
                    mask = (largest_prob>args.confidence_mask).float().detach()
                    loss_cons = loss_cons*mask 
               
                loss_cons = torch.mean(loss_cons)

            else:
                loss_cons = torch.nn.functional.kl_div( 
                         torch.nn.functional.log_softmax(outputs_unlab_aug, dim=1), 
                         torch.nn.functional.softmax(outputs_unlab, dim=1).detach(), reduction='batchmean') 

            loss += args.w_cons*loss_cons
    
        if args.entmin:    
            outputs_unlab, _ = net(inputs_unlab)
            unlab_prob = torch.nn.functional.softmax(outputs_unlab, dim=1)
            unlab_log_prob = torch.log(unlab_prob.detach())
            ent = (-1.0*unlab_prob*unlab_log_prob).sum(1)
            loss_entmin = torch.mean(ent)
            loss += args.w_entmin*loss_entmin

        if args.vat:
            loss_vat = vat_loss(net, inputs_unlab)
            loss += args.w_vat*loss_vat

        if args.pl:
            outputs_unlab, _ = net(inputs_unlab)
            unlab_prob = torch.nn.functional.softmax(outputs_unlab, dim=1)
            largest_prob, largest_index = unlab_prob.max(1)
            mask = (largest_prob>args.pl_conf_mask).float().detach()
            samples_selected +=  mask.sum()
            loss_pl = criterion_pl(outputs_unlab_aug, largest_index.detach())
            loss_pl = loss_pl*mask

            loss_pl = loss_pl.mean()
            loss += args.w_pl*loss_pl

        loss.backward()
        optimizer.step()
        if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
            scheduler.step()
        if args.use_ema:
            ema_net.update(net)
        
        train_loss.update(loss.item())    
        train_loss_lab.update(loss_lab.item())
        
        if args.cons:
            train_loss_cons.update(loss_cons.item())
        if args.rot:
            train_loss_rot.update(loss_rot.item())
        if args.entmin:
            train_loss_entmin.update(loss_entmin.item())
        if args.vat:
            train_loss_vat.update(loss_vat.item())
        if args.pl:
            train_loss_pl.update(loss_pl.item())
        
        if args.verbose:
            progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f | Loss_cons: %.6f | Loss_rot: %.6f | Loss_pl: %.6f'
                % (train_loss.avg, train_loss_lab.avg, train_loss_cons.avg, train_loss_rot.avg, train_loss_pl.avg))

        if i_iter%1000==0:
            test(cycle, i_iter, train_loss, train_loss_lab, train_loss_cons)
       
'''   
def val():
    global best_acc
    net.eval()
    test_loss = AverageMeter()
    correct = 0
    total = 0
    U_all = []
    fp = open('results_with_val.txt','a') 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_val):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
        
            loss = criterion(outputs, targets)
            test_loss.update(loss.item(), inputs.size(0))

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(trainloader_val), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss.avg, 100.*correct/total, correct, total))
'''  

def test(cycle, i_iter, loss, loss_lab, loss_unlab):
    global best_acc
    net.eval()
    test_loss = AverageMeter()
    correct = 0
    total = 0
    U_all = []
    #filename = 'results_semi_' + str(args.batch_size_lab) + '_' + str(args.batch_size_unlab) + '_labels_' +  str(args.num_labeled)  + '_steps_' + str(args.num_steps)+ '_warm_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '.txt'
    filename = os.path.join(CHECKPOINT_DIR, 'results.txt')
    #fp = open('results_semi_64_320_100k_w_flip_and_crop_20k_warm_up_sep_masks_wo_GCN_last_norm_again.txt','a') 
    fp = open(filename, 'a')

    if args.use_ema:
        test_net = ema_net.ema
    else:
        test_net = net

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = test_net(inputs)

            loss = criterion(outputs, targets)
            test_loss.update(loss.item(), inputs.size(0))

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss.avg, 100.*correct/total, correct, total))
     
    fp.write(str(i_iter) + ' ' + str(100.*correct/total) + ' loss: ' +  str(loss) + ' loss lab: ' + str(loss_lab) + ' loss unlab: ' + str(loss_unlab) + '\n')    
   
     
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        state = {
            'net': test_net.state_dict(),
            'acc': acc,
            'cycle': cycle,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.t7')
        torch.save(state, os.path.join(CHECKPOINT_DIR, 'best_ckpt.t7'))
        
        best_acc = acc

    
for cycle in range(args.num_cycles):
    train(cycle, trainloader_lab, trainloader_unlab, scheduler, optimizer)

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

import os
import argparse
import pickle
import numpy as np

#from models.senet import *
#from utils import progress_bar
from augment.cutout import Cutout

#from resnet import resnet18
from wrn import wrn
#from autoaugment_extra_only_color import CIFAR10Policy
from augment.autoaugment_extra import CIFAR10Policy

DATASET = 'CIFAR10'
SEED = 0
SPLIT_ID = None

parse = argparse.ArgumentParser(description='PyTorch SSL CIFAR10 UDA Training')
parse.add_argument('--dataset', type=str, default=DATASET, help='dataset')
parse.add_argument('--num-classes', default=10, type=int, help='number of classes')

parse.add_argument('--lr', default=0.03, type=float, help='learning rate')
parse.add_argument('--softmax-temp', default=-1, type=float, help='softmax temperature controlling')
parse.add_argument('--confidence-mask', default=-1, type=float, help='Confidence value for masking')

parse.add_argument('--num-labeled', default=1000, type=int, help='number of labeled_samples')

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

args = parse.parse_args()

#CHECKPOINT_DIR = './results/dataset_' + str(args.dataset)  + '_labels_' + str(args.num_labeled) + '_batch_lab_' +  str(args.batch_size_lab) + '_batch_unlab_' + str(args.batch_size_unlab)  + '_steps_' + str(args.num_steps) +'_warmup_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '_SEED_' + str(args.seed)
CHECKPOINT_DIR = './results/dataset_' + str(args.dataset) + '_labels_' + str(args.num_labeled) + '_batch_lab_' +  str(args.batch_size_lab) + '_batch_unlab_' + str(args.batch_size_unlab)  + '_steps_' + str(args.num_steps) +'_warmup_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '_SEED_' + str(args.seed)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
np.random.seed(args.seed)

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

# Data
print('==> Preparing data..')
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

transform_train = TransformTwice(transform_ori, transform_aug)
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    labelset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    labelset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
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
mask_ = np.zeros(train_ids.shape[0], dtype=np.bool)
mask_test = np.zeros(test_ids.shape[0], dtype=np.bool)
labels_test = np.array([testset[i][1] for i in test_ids], dtype=np.int64)
for i in range(10):
    mask[np.where(labels == i)[0][: int(args.num_labeled / args.num_classes)]] = True
    mask_[np.where(labels == i)[0][int(args.num_labeled / args.num_classes): ]] = True
    mask_test[np.where(labels_test == i)[0]] = True
labeled_indices, unlabeled_indices = train_ids[mask], train_ids[mask_]
test_indices = test_ids[mask_test]

train_sampler_lab = data.sampler.SubsetRandomSampler(labeled_indices)
train_sampler_unlab = data.sampler.SubsetRandomSampler(unlabeled_indices)
test_sampler = data.sampler.SubsetRandomSampler(test_indices)

trainloader_lab = data.DataLoader(trainset, batch_size=args.batch_size_lab, sampler=train_sampler_lab, num_workers=16, drop_last=True)
trainloader_unlab = data.DataLoader(trainset, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=16, pin_memory=True)

trainloader_val = data.DataLoader(labelset, batch_size=100, sampler=train_sampler_lab, num_workers=16, drop_last=False)

testloader = data.DataLoader(testset, batch_size=100, sampler=test_sampler, num_workers=16)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = wrn(num_classes=args.num_classes).cuda()
#net = models.resnet18().cuda()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, betas= (0.9, 0.999))
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

# Training
def train(cycle, trainloader_lab, trainloader_unlab, scheduler, optimizer):
    print('\nCycle: %d' % cycle)
    train_loss = 0
    train_loss_lab = 0
    train_loss_unlab = 0
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
       
        
        outputs_lab = net(inputs_lab)
        loss_lab = criterion(outputs_lab, targets_lab)
        #inputs, targets = inputs.to(device), targets.to(device)

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

            loss = loss_lab + loss_unlab

            ''' 
            loss_unlab = torch.nn.functional.kl_div( 
                         torch.nn.functional.log_softmax(outputs_unlab_aug/args.softmax_temp, dim=1), 
                         torch.nn.functional.softmax(outputs_unlab/args.softmax_temp, dim=1).detach(), reduction='batchmean') 
            '''

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        train_loss_lab += loss_lab.item()
        if args.lab_only:
            train_loss_unlab = 0.0
        else:
            train_loss_unlab += loss_unlab.item()
        
        #progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f'
            #% (loss.item(), loss_lab.item()))
        if args.verbose:
            progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f | Loss_unlab: %.6f'
                % (train_loss/1000.0, train_loss_lab/1000.0, train_loss_unlab/1000.0))

        if i_iter%1000==0:
            train_loss /= 1000
            train_loss_lab /= 1000
            train_loss_unlab /= 1000
            
            test(cycle, i_iter, train_loss, train_loss_lab, train_loss_unlab)
            #val()        
            
            train_loss = 0
            train_loss_lab = 0
            train_loss_unlab = 0
       
   
def val():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    U_all = []
    fp = open('results_with_val.txt','a') 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_val):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            log_probs = torch.log(probs)*(-1)
            U = (probs*log_probs).sum(1)
            U_all.append(U)
        
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(trainloader_val), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
     

def test(cycle, i_iter, loss, loss_lab, loss_unlab):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    U_all = []
    #filename = 'results_semi_' + str(args.batch_size_lab) + '_' + str(args.batch_size_unlab) + '_labels_' +  str(args.num_labeled)  + '_steps_' + str(args.num_steps)+ '_warm_' + str(args.warm_up_steps) + '_softmax_temp_' + str(args.softmax_temp) + '_conf_mask_' + str(args.confidence_mask) + '.txt'
    filename = os.path.join(CHECKPOINT_DIR, 'results.txt')
    #fp = open('results_semi_64_320_100k_w_flip_and_crop_20k_warm_up_sep_masks_wo_GCN_last_norm_again.txt','a') 
    fp = open(filename, 'a')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            log_probs = torch.log(probs)*(-1)
            U = (probs*log_probs).sum(1)
            U_all.append(U)
        
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
     
    fp.write(str(i_iter) + ' ' + str(100.*correct/total) + ' loss: ' +  str(loss) + ' loss lab: ' + str(loss_lab) + ' loss unlab: ' + str(loss_unlab) + '\n')    
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        state = {
            'net': net.state_dict(),
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

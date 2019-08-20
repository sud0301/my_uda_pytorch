'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
import numpy as np

#from models.senet import *
from utils import progress_bar
from cutout import Cutout

#from data_loader import iCIFAR10
#from resnet import resnet18
from wrn import wrn
from autoaugment_extra import CIFAR10Policy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--lr-warm-up', action='store_true', help='increase lr slowly')
parser.add_argument('--warm-up-steps', default=20000, type=int, help='number of iterations for warmup')

parser.add_argument('--batch-size-lab', default=64, type=int, help='training batch size')
parser.add_argument('--batch-size-unlab', default=320, type=int, help='training batch size')
parser.add_argument('--num-steps', default=100000, type=int, help='number of iterations')

parser.add_argument('--partial-data', default=0.5, type=float, help='partial data')
parser.add_argument("--partial-id", type=str, default=None, help="restore partial id list")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parser.add_argument('--n-holes', default=1, type=float, help='number of holes for cutout')
parser.add_argument('--cutout-size', default=16, type=float, help='size of the cutout window')

parser.add_argument('--autoaugment', action='store_true', help='use autoaugment augmentation')
parser.add_argument('--GCN', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

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
    transforms.ToTensor(),
    #transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_aug = transforms.Compose([
    CIFAR10Policy(),
    transforms.ToTensor(),
    #transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    Cutout(n_holes=args.n_holes, length=args.cutout_size),
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_train = TransformTwice(transform_ori, transform_aug)

#list_classes = [5,6,7,8,9]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
labelset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
#trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

train_dataset_size = len(trainset)
partial_size = int(args.partial_data * train_dataset_size)


if args.partial_id is not None:
    train_ids = pickle.load(open(args.partial_id, 'rb'))
    print('loading train ids from {}'.format(args.partial_id))
else:
    train_ids = np.arange(train_dataset_size)
    np.random.shuffle(train_ids)

pickle.dump(train_ids, open('train_id.pkl', 'wb'))

#train_sampler_lab = data.sampler.SubsetRandomSampler(train_ids[:4000])
#train_sampler_unlab = data.sampler.SubsetRandomSampler(train_ids[4000:])

mask = np.zeros(train_ids.shape[0], dtype=np.bool)
labels = np.array([trainset[i][1] for i in train_ids], dtype=np.int64)
for i in range(10):
    mask[np.where(labels == i)[0][: int(4000 / 10)]] = True
labeled_indices, unlabeled_indices = train_ids[mask], train_ids[~ mask]
#labeled_indices, unlabeled_indices = train_ids[mask], train_ids

train_sampler_lab = data.sampler.SubsetRandomSampler(labeled_indices)
train_sampler_unlab = data.sampler.SubsetRandomSampler(unlabeled_indices)

trainloader_lab = data.DataLoader(trainset, batch_size=args.batch_size_lab, sampler=train_sampler_lab, num_workers=16, drop_last=True)
trainloader_unlab = data.DataLoader(trainset, batch_size=args.batch_size_unlab, sampler=train_sampler_unlab, num_workers=16, pin_memory=True)

trainloader_val = data.DataLoader(labelset, batch_size=100, sampler=train_sampler_lab, num_workers=16, drop_last=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

#trainloader_lab_iter = iter(trainloader_lab)
#trainloader_unlab_iter = iter(trainloader_unlab)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = wrn().cuda()

'''
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.xavier_uniform_(m.weight.data)
        #weights_shape = []
        print ('m size: ', m.size())
        n = m.size(0)*m.size(1)*m.size(3)
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=np.sqrt(2.0/n))
'''
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


def global_contrast_normalize(X, scale=55., min_divisor=1e-8):
    X = X.view(X.size(0), -1)
    X = X - X.mean(dim=1, keepdim=True)

    normalizers = torch.sqrt( torch.pow( X, 2).sum(dim=1, keepdim=True)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers

    return X.view(X.size(0),3,32,32)

def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Training
def train(epoch, trainloader_lab, trainloader_unlab, scheduler, optimizer):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    train_loss_lab = 0
    train_loss_unlab = 0
    correct = 0
    total = 0

    trainloader_lab_iter = iter(trainloader_lab)
    trainloader_unlab_iter = iter(trainloader_unlab)
  
    #net.apply(weights_init) 

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
       
        if args.GCN: 
            inputs_lab = global_contrast_normalize( inputs_lab )
        
        outputs_lab = net(inputs_lab)
        loss_lab = criterion(outputs_lab, targets_lab)
        #inputs, targets = inputs.to(device), targets.to(device)
        
        try:
            batch_unlab = next(trainloader_unlab_iter)
        except:
            trainloader_unlab_iter = iter(trainloader_unlab)
            batch_unlab = next(trainloader_unlab_iter) 
        
        (inputs_unlab, inputs_unlab_aug), _ = batch_unlab
        inputs_unlab, inputs_unlab_aug = inputs_unlab.cuda(), inputs_unlab_aug.cuda()
       
        if args.GCN: 
            inputs_unlab = global_contrast_normalize( inputs_unlab )
            inputs_unlab_aug = global_contrast_normalize( inputs_unlab_aug )

        outputs_unlab = net(inputs_unlab)
        outputs_unlab_aug = net(inputs_unlab_aug)
        #print (targets)
        #loss_unlab = nn.KLDivLoss()(F.log_softmax(outputs_unlab), F.softmax(outputs_unlab_aug))
        #print ('outputs unlab size(): ', outputs_unlab.size())
        #loss_unlab = nn.KLDivLoss()(F.log_softmax(outputs_unlab_aug, dim=1), F.softmax(outputs_unlab, dim=1).detach(), reduction='batchmean')
      
        loss_unlab = torch.nn.functional.kl_div( 
                     torch.nn.functional.log_softmax(outputs_unlab_aug, dim=1), 
                     torch.nn.functional.softmax(outputs_unlab, dim=1).detach(), reduction='batchmean') 
    
        #loss_kldiv = F.kl_div(F.log_softmax(outputs_unlab_aug, dim=1), F.softmax(outputs_unlab, dim=1), reduction='none')    # loss for unsupervised
        #loss_kldiv = torch.sum(loss_kldiv, dim=1)
    
        #loss_unlab =  1.0*torch.mean(loss_kldiv)
 
        loss = loss_lab + loss_unlab

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        train_loss_lab += loss_lab.item()
        train_loss_unlab += loss_unlab.item()

        #progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f'
            #% (loss.item(), loss_lab.item()))
        progress_bar(i_iter, args.num_steps, 'Loss: %.6f | Loss_lab: %.6f | Loss_unlab: %.6f'
            % (train_loss/1000.0, train_loss_lab/1000.0, train_loss_unlab/1000.0))

        if i_iter%1000==0:
            train_loss /= 1000
            train_loss_lab /= 1000
            train_loss_unlab /= 1000
            
            test(epoch, i_iter, train_loss, train_loss_lab, train_loss_unlab)
            val()        
            
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

            progress_bar(batch_idx, len(trainloader_val), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
     

def test(epoch, i_iter, loss, loss_lab, loss_unlab):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    U_all = []
    fp = open('results_semi_64_320_100k_w_flip_and_crop_20k_warm_up_sep_masks_wo_GCN_last_norm_again.txt','a') 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #if args.GCN:
                #inputs = global_contrast_normalize( inputs )
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
     
    fp.write(str(i_iter) + ' ' + str(100.*correct/total) + ' loss: ' +  str(loss) + ' loss lab: ' + str(loss_lab) + ' loss unlab: ' + str(loss_unlab) + '\n')    
    #fp.write(str(i_iter))    
    
    #U_all = torch.cat(U_all, dim=0)
    #U_sorted, _ = U_all.sort()
    #print ('average entropy: ', U_all.float().mean())
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        '''
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        '''
        best_acc = acc
    

for epoch in range(start_epoch, start_epoch+100):
    train(epoch, trainloader_lab, trainloader_unlab, scheduler, optimizer)
    if epoch%5==0:
        test(epoch)


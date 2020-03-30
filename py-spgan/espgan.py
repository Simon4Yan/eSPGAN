#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
from models.models import ft_net
import models.models_spgan as models
import torch
from torch import nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.utils as utils
from utils.data_loader_reid import get_loader
from PIL import Image
from utils.logger import Logger
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time
import datetime
from torch.nn.parallel import DataParallel


#gpu 
# Device configuration
device = torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')

# params 
epochs = 15
batch_size = 8
lr = 0.0002
lambda1 = 10.0
lambda2 = 5.0
lambda3 = 2.0
lambda4 = 5.0 # for IDE
margin = 2.0
mode = 'train'
num_workers = 0
n_ide = 0
class_num = 751
use_tensorboard = True
if use_tensorboard:
    log_dir = './checkpoints/espgan_m2d_lam5/'
    utils.mkdir(log_dir)
    Logger = Logger(log_dir)
	
# data 
image_size = 286
Market_crop_size = 256
Duke_crop_size = 256
Duke_image_dir = '../weijian/pybaseline/datasets/duke/pytorch/train_all/'
Market_image_dir = '../weijian/pybaseline/datasets/market/pytorch/train_all/'

Duke_loader = get_loader(Duke_image_dir,
                         Duke_crop_size, image_size, batch_size,
                           'Duke', mode, num_workers)
Market_loader = get_loader(Market_image_dir,
                         Market_crop_size, image_size, batch_size,
                           'Market', mode, num_workers)
                                               
b_loader = Duke_loader
a_loader = Market_loader                                                    
                           
a_fake_pool = utils.ItemPool()
b_fake_pool = utils.ItemPool()

# model
# For IDE 
IDE = ft_net(class_num).to(device)
IDE.load_state_dict(torch.load('./models/IDE_market.pth', map_location=torch.device('cpu'))) 

Da = DataParallel(models.Discriminator(), device_ids=[0,1]).to(device)
Db = DataParallel(models.Discriminator(), device_ids=[0,1]).to(device)
Ga = DataParallel(models.Generator(), device_ids=[0,1]).to(device)
Gb = DataParallel(models.Generator(), device_ids=[0,1]).to(device)
MSE = nn.MSELoss()
L1 = nn.L1Loss()
def classification_loss (logit, target):
    """Compute softmax cross entropy loss."""
    return F.cross_entropy(logit, target)
    
da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))

IDE_criterion = torch.nn.CrossEntropyLoss()     
ignored_params = list(map(id, IDE.model.fc.parameters() )) + list(map(id, IDE.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, IDE.parameters())
# Observe that all parameters are being optimized
IDE_optimizer = torch.optim.SGD([
     {'params': base_params, 'lr': 0.001},
     {'params': IDE.model.fc.parameters(), 'lr': 0.01},
     {'params': IDE.classifier.parameters(), 'lr': 0.01}
 ], momentum=0.9, weight_decay=5e-4, nesterov=True)
# Decay LR by a factor of 0.1 every 20 epochs (20 epochs for market and 30 epochs for duke)
scheduler_IDE = lr_scheduler.StepLR(IDE_optimizer, step_size=10, gamma=0.1)


## load checkpoint 
ckpt_dir = './checkpoints/espgan_m2d_lam5/'
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir, map_location=torch.device('cpu'))
    start_epoch = ckpt['epoch']
    Da.load_state_dict(ckpt['Da'])
    Db.load_state_dict(ckpt['Db'])
    Ga.load_state_dict(ckpt['Ga'])
    Gb.load_state_dict(ckpt['Gb'])
    IDE.load_state_dict(ckpt['IDE'])

    da_optimizer.load_state_dict(ckpt['da_optimizer'])
    db_optimizer.load_state_dict(ckpt['db_optimizer'])
    ga_optimizer.load_state_dict(ckpt['ga_optimizer'])
    gb_optimizer.load_state_dict(ckpt['gb_optimizer'])
    IDE_optimizer.load_state_dict(ckpt['IDE_optimizer'])
except:
    start_epoch = 0
    print('Training form zero')    

## run 
# Start training.
print('Start training...')
start_time = time.time()
for epoch in range(start_epoch, epochs):
    scheduler_IDE.step()
    for i, (a_real, b_real) in enumerate(itertools.izip(a_loader, b_loader)):
        # step
        step = epoch% min(len(a_loader), len(b_loader)) + i + 1

        # set train
        Ga.train()
        Gb.train()

        # leaves
        label_a = a_real[1].to(device)
        a_real = a_real[0].to(device)
        b_real = b_real[0].to(device)
        
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)
        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        loss = {}
        # =================================================================================== #
        #                               1. Train the Generator (Ga and Gb)                    #
        # =================================================================================== #
        if i % 3 ==0:
            
            # identity loss
            b2b = Gb(b_real)
            a2a = Ga(a_real)
            idt_loss_b = L1(b2b, b_real)
            idt_loss_a = L1(a2a, a_real)
            idt_loss = idt_loss_a + idt_loss_b
            
            # gen losses
            a_f_dis = Da(a_fake)
            b_f_dis = Db(b_fake)
            r_label = torch.ones(a_f_dis.size()).to(device)
            a_gen_loss = MSE(a_f_dis, r_label)
            b_gen_loss = MSE(b_f_dis, r_label)
    
            # rec losses
            a_rec_loss = L1(a_rec, a_real)
            b_rec_loss = L1(b_rec, b_real)
            rec_loss = a_rec_loss + b_rec_loss
            
            # ide loss for G  
            inputs_G = torch.cat((b_fake, a_rec), dim=0)
            label_G = torch.cat((label_a,label_a), dim=0).to(device)
            
            # forward
            IDE.to(device)
            IDE.eval()
            outputs = IDE(inputs_G)
            _, preds = torch.max(outputs, 1)
            G_ide_loss = classification_loss(outputs, label_G)
            G_batch_acc = (torch.sum(preds == label_G).item()) / preds.size(0)
            
            # g loss			
            g_loss = a_gen_loss + b_gen_loss + lambda1*rec_loss  + lambda4 *G_ide_loss + lambda2*idt_loss

            loss['Ga_gen_loss'] = a_gen_loss.item()
            loss['Gb_gen_loss'] = b_gen_loss.item()
            loss['Grec_loss'] = rec_loss.item()
            loss['G_ide_loss'] = G_ide_loss.item()
            loss['G_batch_acc'] = G_batch_acc
            
            Ga.zero_grad()
            Gb.zero_grad()   
            g_loss.backward()
            ga_optimizer.step()
            gb_optimizer.step()
        # =================================================================================== #
        #                               2. Train IDE for re-ID                                #
        # =================================================================================== #
        # Fix the bn of basemodel
        IDE.to(device)
        IDE.eval()
        IDE.model.fc.train(True)
        # Input data and label for IDE 
        #(a half of data are real images and another half of data are generated images)
        try:
            x_real_ide, label_ide = next(data_iter_source)
        except:
            data_iter_source = iter(a_loader)
            x_real_ide, label_ide = next(data_iter_source) 
        x_real_ide = x_real_ide.to(device)
        x_fake_s = b_fake.detach()
        inputs = torch.cat((x_real_ide, x_fake_s), dim=0)
        label = torch.cat((label_ide.to(device), label_a), dim=0)
        # forward
        outputs = IDE(inputs)
        _, preds = torch.max(outputs, 1)
        ide_loss_cls = classification_loss(outputs, label)
        batch_acc = (torch.sum(preds == label).item()) / preds.size(0)

        # Backward and optimize.
        IDE_optimizer.zero_grad()
        ide_loss_cls.backward()
        IDE_optimizer.step()

        # Logging.
        loss['IDE/ide_loss_cls'] = ide_loss_cls.item()
        loss['IDE/batch_acc'] = batch_acc    
        # =================================================================================== #
        #                               3. Train the Discriminator                            #
        # =================================================================================== #        
        # leaves
        a_fake = torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0]).to(device)
        b_fake = torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0]).to(device)
        
        ## Training D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = torch.ones(a_f_dis.size()).to(device)
        f_label = torch.zeros(a_f_dis.size()).to(device)

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = (a_d_r_loss + a_d_f_loss)*0.5
        b_d_loss = (b_d_r_loss + b_d_f_loss)*0.5
        
        loss['Da_d_f_loss'] = a_d_f_loss.item()
        loss['Db_d_f_loss'] = b_d_f_loss.item()
        loss['Da_d_r_loss'] = a_d_r_loss.item()
        loss['Db_d_r_loss'] = b_d_r_loss.item()		
        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()
        #==================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if i % 10 == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Epoch [{}], Iteration [{}/{}])".format(et, epoch, i, min(len(a_loader), len(b_loader)))
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)
            if use_tensorboard:
                for tag, value in loss.items():
                     Logger.scalar_summary(tag, value, i) 
        if i % 50 == 0:
            with torch.no_grad():
                Ga.eval()
                Gb.eval()  
                a_real_test =(iter(a_loader).next()[0]).to(device)
                b_real_test = (iter(b_loader).next()[0]).to(device)
                # train G
                a_fake_test = Ga(b_real_test)
                b_fake_test = Gb(a_real_test)
    
                a_rec_test = Ga(b_fake_test)
                b_rec_test = Gb(a_fake_test)
    
                pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1)/2.0
    
                save_dir = './sample_images_while_training/espgan_m2d_lam5/'
                utils.mkdir(save_dir)
                torchvision.utils.save_image(pic, '%sEpoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=batch_size)
    utils.save_checkpoint({'epoch':epoch + 1,
                           'Da': Da.state_dict(),
                           'Db': Db.state_dict(),
                           'Ga': Ga.state_dict(),
                           'Gb': Gb.state_dict(),
                           'IDE': IDE.state_dict(),
                           'da_optimizer': da_optimizer.state_dict(),
                           'db_optimizer': db_optimizer.state_dict(),
                           'ga_optimizer': ga_optimizer.state_dict(),
                           'gb_optimizer': gb_optimizer.state_dict(),
                           'IDE_optimizer': IDE_optimizer.state_dict()},
                          '%sEpoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=6)

from __future__ import print_function, division

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchsummary
import torchvision

from PIL import Image

from Data_Loader import Images_Dataset, Images_Dataset_folder
from isic import ISIC
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, Resnet_Unet, AttU_Net_Ds
from Models2 import reS_Unet
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
from utils import LR_Scheduler

import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil


'''
  对原有的训练文件的重写
  暂不完全重写，仅满足deepsupervision的训练
'''


#########################################################
## setting the devices to be used
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available. Training on CPU')
else:
    device = torch.device("cuda:0")
    print('CUDA is available. Training on GPU')

#device = torch.device("cuda:0" if train_on_gpu else "cpu")

##############################################################################
## setting the parameters of the model
shuffle = True
batch_size = 4
print('batch_size = ' + str(batch_size))
valid_size = 0.15
epoch = 50
print('epoch = ' + str(epoch))
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True




#######################################################################
## setting the dataset information
##### path info
train_imgs = '/home/dxw/Dataset/ISIC2017_train_imgs/'
train_labels = '/home/dxw/Dataset/ISIC2017_train_labels/'
validation_imgs = '/home/dxw/Dataset/ISIC2017_val_imgs/'
validation_labels = '/home/dxw/Dataset/ISIC2017_val_labels/'
test_imgs = '/home/dxw/Dataset/ISIC2017_test_imgs/*'
test_labels = '/home/dxw/Dataset/ISIC2017_test_labels/*'
test_image = '/home/dxw/Dataset/ISIC2017_test_imgs/ISIC_0012086.jpg'
test_label = '/home/dxw/Dataset/ISIC2017_test_labels/ISIC_0012086_segmentation.png'

#Training_Data = Images_Dataset_folder(train_imgs, train_labels)
#Validation_Data = Images_Dataset_folder(validation_imgs, validation_labels)
Training_Data = ISIC(train_imgs, train_labels, applyAutoAug = True, 
    applyCutout = False, crop_size=( 192, 256),base_size=( 192, 256), multi_scale=False,)
Validation_Data = ISIC(validation_imgs, validation_labels, applyAutoAug = False,
    applyCutout = False, crop_size=( 192, 256),base_size=( 192, 256), num_classes=2, 
    multi_scale=False, flip=False,)


num_train = len(Training_Data)
num_valid = len(Validation_Data)
indices_train = list(range(num_train))
indices_valid = list(range(num_valid))

if shuffle:
    np.random.seed(random_seed)
    #np.random.shuffle(indices)
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_valid)

#train_idx, valid_idx = indices[split:], indices[:split]
train_idx, valid_idx = indices_train, indices_valid
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)
valid_loader = torch.utils.data.DataLoader(Validation_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)


###############################################################################
## model definition
model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Resnet_Unet, reS_Unet, AttU_Net_Ds]

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

##### passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary
model_test = model_unet(model_Inputs[7], 3, 1)
model_test.to(device)
##### show the summary of the model
#torchsummary.summary(model_test, input_size=(3, 128, 128))


#### optimization parameters
initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
#opt = torch.optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.90)
MAX_STEP = int(500)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-8)
scheduler = LR_Scheduler('cos', initial_lr, epoch, int(2000/batch_size))
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=8, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
#
#https://pytorch.org/docs/stable/optim.html
#args.lr_scheduler:'cos', 'poly'
#scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))
#


############################################################
## creation of related folders
#### Creating a Folder for every data of the program
New_folder = './model'
if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)
try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#### Setting the folder of saving the predictions
read_pred = './model/pred'
if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)
try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#### checking if the model exists and if true then delete
read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)
if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')
try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

###############################################################
## some data-transform
data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
data_transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
           torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
#           torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])



#############################################################
## training procedure
print('######################################')
for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    #scheduler.step(i)
    #lr = scheduler.get_lr()

    #######################################################
    #Training Data
    #######################################################

    model_test.train()
    #k = 1
    c = 0
    print('start training----------trainloader')
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        #If want to get the input images with their Augmentation - To check the data flowing in net
        #input_images(x, y, i, n_iter, k)

       # grid_img = torchvision.utils.make_grid(x)
        #writer1.add_image('images', grid_img, 0)

       # grid_lab = torchvision.utils.make_grid(y)

       # new scheduler -----  
       # scheduler(self.optimizer, i, epoch, self.best_pred)

        opt.zero_grad()

        ### single loss implementation
        #y_pred = model_test(x)
        #lossT = calc_loss(y_pred, y)     # Dice_loss Used

        
        
        ### supervision training implementation
        pred5, pred4, pred3, pred = model_test(x)
        y3 = F.interpolate(y, scale_factor=0.5)
        #print("inside train")

        y4 = F.interpolate(y, scale_factor=0.25)
        y5 = F.interpolate(y, scale_factor=0.125)
        #print(y3.shape)
        #print(y4.shape)
        #print(y5.shape)
        lossMain = calc_loss(pred, y)
        loss3 = calc_loss(pred3, y3)
        loss4 = calc_loss(pred4, y4)
        loss5 = calc_loss(pred5, y5)
        ### weight parameters need to be finetune
        ### apply a weight decay strategy !!! ??
        lossT = 0.35*lossMain+0.25*loss3+0.25*loss4+0.15*loss5
        

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
      #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()
        x_size = lossT.item() * x.size(0)
        scheduler(opt, c, i)
        c = c+1
        #k = 2


    #    for name, param in model_test.named_parameters():
    #        name = name.replace('.', '/')
    #        writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
    #        writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)


    #### Validation Step
    print('start validating----------valid_loader')
    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory
    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)

        pred5, pred4, pred3, pred = model_test(x)
        #y3 = F.interpolate(y, scale_factor=0.5)
        #y4 = F.interpolate(y, scale_factor=0.25)
        #y5 = F.interpolate(y, scale_factor=0.125)
        lossMain = calc_loss(pred, y)
        #loss3 = calc_loss(pred3, y3)
        #loss4 = calc_loss(pred4, y4)
        #loss5 = calc_loss(pred5, y5)
        ### weight parameters need to be finetune
        ### apply a weight decay strategy !!! ??
        #lossL = 0.25*lossMain+0.25*loss3+0.25*loss4+0.25*loss5
        
        valid_loss += lossMain.item() * x1.size(0)
        x_size1 = lossMain.item() * x1.size(0)

    ### save a prediction
    print('start saving ------------')
    im_tb = Image.open(test_image)
    im_label = Image.open(test_label)
    s_tb = data_transform(im_tb)
    s_label = data_transform2(im_label)
    s_label = s_label.detach().numpy()
    ## make a prediction
    pred_tb = model_test(s_tb.unsqueeze(0).to(device))[-1].cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()
    #pred_tb = threshold_predictions_v(pred_tb)
    x1 = plt.imsave(
        './model/pred/img_iteration_' + str(n_iter) + '_epoch_'
        + str(i) + '.png', pred_tb[0][0])
    ## can insert some measure metrics here
  #  accuracy = accuracy_score(pred_tb[0][0], s_label)

    #######################################################
    
    #######################################################
    # show the loss value
    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)
    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss, valid_loss))
    #To write in Tensorboard
 #       writer1.add_scalar('Train Loss', train_loss, n_iter)
  #      writer1.add_scalar('Validation Loss', valid_loss, n_iter)
        #writer1.add_image('Pred', pred_tb[0]) #try to get output of shape 3


    #######################################################
    #Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./model/Unet_D_' +
                                              str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
       # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss
        #if i_valid ==3:
         #   break

    #######################################################
    # Extracting the intermediate layers
    #######################################################

    #####################################
    # for kernals
    #####################################
    x1 = torch.nn.ModuleList(model_test.children())
    # x2 = torch.nn.ModuleList(x1[16].children())
     #x3 = torch.nn.ModuleList(x2[0].children())

    #To get filters in the layers
     #plot_kernels(x1.weight.detach().cpu(), 7)

    #####################################
    # for images
    #####################################
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer

    img = Image.open(test_image)
    s_tb = data_transform(img)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device))[-1].cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    plot_kernels(dr.features, n_iter, 7, cmap="rainbow")

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

#######################################################
#closing the tensorboard writer
#######################################################

#writer1.close()

#######################################################
#if using dict
#######################################################

#model_test.filter_dict



from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

##
## 在原有数据类中进行推理及保存并没有
##进行相应的实现
##

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, Resnet_Unet, AttU_Net_Ds
from Models2 import reS_Unet
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time



test_on_gpu = torch.cuda.is_available()

if not test_on_gpu:
	device = torch.device("cpu")
	print('CUDA is not available. Testing on CPU')
else:
	device = torch.device("cuda:0")
	print('CUDA is available. Testing on GPU')


batch_size = 4
epoch = 50
print('batch_size = ' + str(batch_size))
#valid_size = 0.15
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
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
if test_on_gpu:
    pin_memory = True


model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Resnet_Unet, reS_Unet, AttU_Net_Ds]

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary
model_test = model_unet(model_Inputs[7], 3, 1)
model_test.to(device)


t_data = '/home/dxw/Dataset/ISIC2017_train_imgs/'
l_data = '/home/dxw/Dataset/ISIC2017_train_labels/'
test_image = '/home/dxw/Dataset/ISIC2017_test_imgs/ISIC_0012086.jpg'
test_label = '/home/dxw/Dataset/ISIC2017_test_labels/ISIC_0012086_segmentation.png'
test_folderP = '/home/dxw/Dataset/ISIC2017_test_imgs/'
test_folderL = '/home/dxw/Dataset/ISIC2017_test_labels/'
val_folderP = '/home/dxw/Dataset/ISIC2017_val_imgs/'
val_folderL = '/home/dxw/Dataset/ISIC2017_val_labels/'


## load model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    pin_memory = True

model_test.load_state_dict(torch.load('./model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))
model_test.eval()


### create related folders
read_test_folder112 = './model/gen_images'
if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './model/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './model/label_threshold'

if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)
try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)

## need modifications
Validation_Data = ISIC(validation_imgs, validation_labels, applyAutoAug = False,
    applyCutout = False, crop_size=( 192, 256),base_size=( 192, 256), num_classes=2, 
    multi_scale=False, flip=False,)
valid_loader = torch.utils.data.DataLoader(Validation_Data, batch_size=1, 
                               num_workers=4, pin_memory=pin_memory,)

for x1, y1 in valid_loader:
	x1, y1 = x1.to(device), y1.to(device)
	## base_dataset inference need to be changed!
	pred = Validation_Data.multi_scale_inference(self, model_test, x1, scales=[1], flip=False)
	## need to change the save part in the isic dataset


	#pred5, pred4, pred3, pred = model_test(x)
	#y3 = F.interpolate(y, scale_factor=0.5)
	#y4 = F.interpolate(y, scale_factor=0.25)
	#y5 = F.interpolate(y, scale_factor=0.125)
    #lossMain = calc_loss(pred, y)
    #loss3 = calc_loss(pred3, y3)
    #loss4 = calc_loss(pred4, y4)
    #loss5 = calc_loss(pred5, y5)
    ### weight parameters need to be finetune
    ### apply a weight decay strategy !!! ??
    #lossL = 0.25*lossMain+0.25*loss3+0.25*loss4+0.25*loss5
    #valid_loss += lossMain.item() * x1.size(0)
    #x_size1 = lossMain.item() * x1.size(0)





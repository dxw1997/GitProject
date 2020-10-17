import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from base_dataset import BaseDataset
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from cutout import Cutout


class ISIC(BaseDataset):
    def __init__(self, 
                 images_dir, 
                 labels_dir, 
                 applyAutoAug = False,
                 applyCutout = False,
                 crop_size=( 192, 256),
                 base_size=( 384, 512),
                 num_classes=2, 
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 downsample_rate=1,
                 scale_factor=15,
                 mean=[0.7077172, 0.5913799, 0.54669064], 
                 std=[0.15470739, 0.16332993, 0.17838475]):


#   normMean = [0.7077172, 0.5913799, 0.54669064]
#   normStd = [0.15470739, 0.16332993, 0.17838475]

        
        
        super(ISIC, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        self.applyAutoAug = applyAutoAug
        self.applyCutout = applyCutout
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def __len__(self):

        return len(self.images)


    def __getitem__(self, i):

        image = cv2.imread(self.images_dir + self.images[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.base_size, interpolation = cv2.INTER_LINEAR)
        if self.applyAutoAug:
            image = Image.fromarray(image.astype('uint8'))
            #policy = ImageNetPolicy()
            policy = CIFAR10Policy()
            #policy = SVHNPolicy()
            image = policy(image)
            image = np.array(image)
            #img = np.asarray(image)
        if self.applyCutout:
            image = Cutout(image)##这么写是错的
        #size = image.shape
        ##normalize and change it to a tensor
        #image = self.input_transform(image)
        #image = image.transpose((2, 0, 1))

        label = cv2.imread(self.labels_dir + self.labels[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, self.base_size, interpolation = cv2.INTER_NEAREST)
        
        #some operations needed here
        ## depends on the range of label values
        #label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
        label = np.expand_dims(label, axis=0)

        return image.copy(), label.copy()

    ## the part depends on the deal of the original image sizes
    ## may need to be changed !
    ## /(len(scales))??
    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))


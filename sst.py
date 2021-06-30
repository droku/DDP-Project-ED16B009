from __future__ import print_function, division
from byol_pytorch import BYOL
from torchvision import models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import time
import os
import copy

from PIL import Image
import cv2
import kornia
import albumentations as A
from skimage import io
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import log_loss,roc_curve,accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained=True)

to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((224,224)),
    #transforms.RandomRotation(50),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

data_path = "images3/train/"     
valid_path = "images3/val/"

train_list = os.listdir(data_path)
valid_list = os.listdir(valid_path)
print("train_images: ",len(train_list))
print("val_images: ", len(valid_list))
class train_data_generator(Dataset):
    def __init__(self, image_ls):
        self.is_test = True
        self.image_ls = image_ls 

    def __len__(self):
        return len(self.image_ls)
 
    def __getitem__(self, index):
        if self.is_test:
            id_ = self.image_ls[index]
            image_mask_path = "images3/train/" + id_
            image = cv2.imread(image_mask_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = to_tensor(image)
            return image
class valid_data_generator(Dataset):
    def __init__(self, image_ls):
        self.is_test = True
        self.image_ls = image_ls 

    def __len__(self):
        return len(self.image_ls)
 
    def __getitem__(self, index):
        if self.is_test:
            id_ = self.image_ls[index]
            image_mask_path = "images3/val/" + id_
            image = cv2.imread(image_mask_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = to_tensor(image)
            return image

#transform_train = transforms.Compose([transform.RandomHorizontalFlip(),transforms.to_tensor(),transforms.Normalize(rgb_mean, rgb_std)])

train_dataset = train_data_generator(train_list)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=False)

valid_dataset = valid_data_generator(valid_list)
valid_dataloader = DataLoader(dataset = valid_dataset, batch_size=20, shuffle=False)                                   


learner = BYOL(
    vgg,
    image_size = 224,
    hidden_layer = 'avgpool',
    use_momentum = False       # turn off momentum in the target encoder
)
learner = learner.to(device)

augment_fn = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip()
)

opt = optim.Adam(learner.parameters(), lr=0.000001)
dataloaders = {'train':train_dataloader,'val':valid_dataloader}

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

def train_model(model, optimizer, dataloaders,epochs=25, is_train=True):
	print("no of epochs: ", epochs)
	best_model_loss = copy.deepcopy(model.state_dict())
	best_loss = 1000000.0
	tr_ls, vl_ls = [], []
	
	for epoch in range(epochs):
		print("EPOCH: ", epoch)

		for phase in ['train', 'val']:
			running_loss = 0.0
			for image_train in dataloaders[phase]:
				image_train = image_train.to(device)
				opt.zero_grad()

				with torch.set_grad_enabled(phase=="train"):
					loss = model(image_train)

					if phase == 'train':
						loss.backward()
						opt.step()
				running_loss+= loss.item()*image_train.size(0)
			print("running loss: ",running_loss)
			print("function_loss: ",loss)

			if phase == 'train':
				tr_ls.append(running_loss)
			else:
				vl_ls.append(running_loss)

		if phase == 'val' and running_loss < best_loss:
			best_loss = running_loss
			best_model_loss = copy.deepcopy(model.state_dict())
	print("Training Complete")
	print("best loss: ", best_loss)
	model.load_state_dict(best_model_loss)
	torch.save(model.state_dict(), "./vgg_model.pt")
	
	N = np.arange(len(tr_ls))
	N_v = np.arange(len(vl_ls))
	print(tr_ls)
	print("validation loss: ",vl_ls)
	print("numbers: ",N_v)
	plt.figure()
	plt.plot(N, tr_ls, label="train_loss")
	plt.plot(N, vl_ls[:len(N)], label="val_loss")
	plt.title("Training loss and validation Loss [Epoch{}]".format(epoch))
	plt.xlabel("Epoch #")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig("vgg_model_Training_loss_graph.png")
	plt.close()
	

	return model, best_loss


final_model, best_loss_model = train_model(learner, opt, dataloaders,200)

print("Finished execution")


# save your improved network

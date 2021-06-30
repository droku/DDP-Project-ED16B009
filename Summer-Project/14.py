from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import time
import os
import copy

from PIL import Image
import cv2
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
 
def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
 
to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
# size = 380
size = 224
 
class data_generator(Dataset):
    def __init__(self, image_ls):
        self.image_ls = image_ls 

    def __len__(self):
        return len(self.image_ls)
 
    def __getitem__(self, index):
        ID = str(self.image_ls[index][1])
        image_mask_path, label = "data/clahe/"+str(size)+"/"+ str(self.image_ls[index][1]), self.image_ls[index][2]
        image = cv2.imread(image_mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = to_tensor(image)
        return ID,image, label

train_df= pd.read_csv('t14_aug.csv',index_col=None, header=0)
valid_df= pd.read_csv('v14_aug.csv',index_col=None, header=0)

class_0 = train_df[train_df["Disease_Risk"]==0]
class_1 = train_df[train_df["Disease_Risk"]==1]
class_0_ls = class_0.values.tolist()
class_1_ls = class_1.values.tolist()
random.shuffle(class_0_ls) 
random.shuffle(class_1_ls) 

print(len(class_1),len(class_0))
print(class_1_ls[:10])

train_ls = []
for i in range(len(class_1_ls)):
    train_ls.append(class_1_ls[i])
    train_ls.append(class_0_ls[i%len(class_0)])

valid_ls = valid_df.values.tolist()
random.shuffle(train_ls) 
random.shuffle(valid_ls) 
 
print(len(train_ls))
print(len(valid_ls))

train_flow = data_generator(train_ls)
valid_flow = data_generator(valid_ls)

EPOCHS = 400
BATCH_SIZE = 18
train_iterator = DataLoader(train_flow, batch_size=BATCH_SIZE, shuffle=False)
valid_iterator = DataLoader(valid_flow, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = {'train':train_iterator,'val':valid_iterator}
dataset_sizes = {'train':len(train_ls),'val':len(valid_ls)}
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts_acc = copy.deepcopy(model.state_dict())
    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_loss, best_roc_score = 1000000.0,0.0
    tr_ls, vl_ls, tr_acc, vl_acc  = [],[],[],[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            test_labels,predict_test = [],[]
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            true_ls,pred_ls = [],[]
            for ID, inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)  

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print("outputs",outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                test_labels.extend(labels.data.cpu().numpy())
                predict_test.extend(preds.data.cpu().numpy())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_roc_score = roc_auc_score(test_labels,predict_test)
            if phase=="train":
                tr_ls.append(epoch_loss)
                tr_acc.append(epoch_roc_score)
            elif phase=="val":
                vl_ls.append(epoch_loss)
                vl_acc.append(epoch_roc_score)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print(confusion_matrix(test_labels,predict_test))
            print("roc_score",epoch_roc_score)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts_loss = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "saved_plts_wts/14/loss.pt")
            if phase == 'val' and epoch_roc_score > best_roc_score:
                best_roc_score = epoch_roc_score
                best_model_wts_acc = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "saved_plts_wts/14/roc.pt")


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best roc_score: {:4f}'.format(best_roc_score))
    df = pd.DataFrame(tr_ls, columns = ['train_loss'])  
    df.to_csv('saved_plts_wts/14/train_loss.csv', index=False)
    df = pd.DataFrame(vl_ls, columns = ['val_loss'])  
    df.to_csv('saved_plts_wts/14/val_loss.csv', index=False)
    df = pd.DataFrame(vl_acc, columns = ['val_roc'])  
    df.to_csv('saved_plts_wts/14/val_roc.csv', index=False)
    N = np.arange(0, len(tr_ls))
    plt.figure()
    plt.plot(N, tr_ls, label = "train_loss")
    plt.plot(N, vl_ls, label = "val_loss")
    plt.title(" Training Loss and Validation Loss [Epoch {}]".format(epoch))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    # plt.ylim([0,1])
    plt.legend()
    #plt.savefig('g1/LossEpoch-{}.png'.format(epoch))
    plt.savefig('saved_plts_wts/14/LossEpoch.png')
    plt.close()
    plt.figure()
    # plt.plot(N, tr_acc, label = "train_acc")
    plt.plot(N, vl_acc, label = "val_acc")
    plt.title("Validation Accuracy [Epoch {}]".format(epoch))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig('g1/3AccuracyEpoch-{}.png'.format(epoch))
    plt.savefig('saved_plts_wts/14/AccuracyEpoch.png')
    plt.close()
    return best_model_wts_loss, best_model_wts_acc
 
model_conv = torchvision.models.resnet34(pretrained=True)
#for param in model_conv.parameters():
#    param.requires_grad = False
# model_conv = torchvision.models.inception_v3(pretrained=True)
# model_conv.aux_logits = False
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_ft = model_conv.to(device)

# weights = [0.9,0.1]
weights = [0.6,0.4]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)
# exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer_ft, base_lr=0.00001,
#  max_lr=0.01, step_size_up=10, step_size_down=10)
loss_wts,acc_wts  = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=30)

to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
 
print("WEIGHTS LOADED")

id_vals = ["_cl_10_.png","_cl_10_vf_.png","_cl_10_hf_.png","_cl_10_vhf_.png",
"_cl_20_.png","_cl_20_vf_.png","_cl_20_hf_.png","_cl_20_vhf_.png",]

test_ls = os.listdir("Evaluation_Set/")
test_ls = sorted(test_ls,key=lambda x:int(x[:-4]))   
pred_sample= pd.read_csv("teamName_results.csv")  
col = list(pred_sample.columns) 

test_dic = {}
for i in range(len(test_ls)):
    curr = test_ls[i]
    same = []
    for c in range(len(id_vals)):
        cnhg = curr[:-4]+id_vals[c]
        same.append(cnhg)
    test_dic[curr] = same

# size = 512
model_ft.load_state_dict(torch.load("saved_plts_wts/14/loss.pt"))
print("LOSS WEIGHTS LOADED")
pred_ls = []
model_ft.eval()
with torch.no_grad():
    for i in range(len(test_ls)):
        curr_res = []
        ID = int(test_ls[i][:-4])
        curr_res.append(ID)
        cnt0,cnt1,c0,c1,out0,out1 = 0,0,0,0,0,0
        img_aug = test_dic[test_ls[i]]
        for i in range(len(img_aug)):
            inp = img_aug[i]
            image_mask_path = "Eval/clahe/"+str(size)+"/"+ inp
            image = cv2.imread(image_mask_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = to_tensor(image)
            image = torch.unsqueeze(image,dim=0)
            image = image.to(device)
            outputs = model_ft(image)
            outputs = F.softmax(outputs,dim=1)
            # print("outputs",outputs)
            outputs = outputs.data.cpu().numpy()[0]
            top_pred = outputs.argmax()
            if top_pred==0:
                cnt0+=1
                out0 = max(out0,outputs[top_pred])
            else:
                cnt1+=1
                out1 = max(out1,outputs[top_pred])
            # if outputs[0]>0.5:
            #     cnt1+=1
            #     c1+=outputs[0]-0.5
            # else:
                # pred_ls.append(0)
                # cnt0+=1
                # c0+=0.5-outputs[0]
            # true_ls.append(label)
        if cnt1>4:
            curr_res.append(out1)
            # curr_res.append(1)
        elif cnt0>4:
            # curr_res.append(0)
            curr_res.append(1-out0)
        elif c1>c0:
            curr_res.append(out1)
            # curr_res.append(1)
        else:
            # curr_res.append(0)
            curr_res.append(1-out0)
        pred_ls.append(curr_res)
df = pd.DataFrame(pred_ls, columns = ['ID', 'Disease_Risk'])  
for c in col[2:]:
    df[c] = 0
df.to_csv('saved_plts_wts/14/mirl_results_loss.csv', index=False)
print("LOSS DONE")

model_ft.load_state_dict(torch.load("saved_plts_wts/14/roc.pt"))
print("ROC WEIGHTS LOADED")
pred_ls = []
model_ft.eval()
with torch.no_grad():
    for i in range(len(test_ls)):
        curr_res = []
        ID = int(test_ls[i][:-4])
        curr_res.append(ID)
        cnt0,cnt1,c0,c1,out0,out1 = 0,0,0,0,0,0
        img_aug = test_dic[test_ls[i]]
        for i in range(len(img_aug)):
            inp = img_aug[i]
            image_mask_path = "Eval/clahe/"+str(size)+"/"+ inp
            image = cv2.imread(image_mask_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = to_tensor(image)
            image = torch.unsqueeze(image,dim=0)
            image = image.to(device)
            outputs = model_ft(image)
            outputs = F.softmax(outputs,dim=1)
            # print("outputs",outputs)
            outputs = outputs.data.cpu().numpy()[0]
            top_pred = outputs.argmax()
            if top_pred==0:
                cnt0+=1
                out0 = max(out0,outputs[top_pred])
            else:
                cnt1+=1
                out1 = max(out1,outputs[top_pred])
            # if outputs[0]>0.5:
            #     cnt1+=1
            #     c1+=outputs[0]-0.5
            # else:
                # pred_ls.append(0)
                # cnt0+=1
                # c0+=0.5-outputs[0]
            # true_ls.append(label)
        if cnt1>4:
            curr_res.append(out1)
            # curr_res.append(1)
        elif cnt0>4:
            # curr_res.append(0)
            curr_res.append(1-out0)
        elif c1>c0:
            curr_res.append(out1)
            # curr_res.append(1)
        else:
            # curr_res.append(0)
            curr_res.append(1-out0)
        pred_ls.append(curr_res)
df = pd.DataFrame(pred_ls, columns = ['ID', 'Disease_Risk'])  
for c in col[2:]:
    df[c] = 0
df.to_csv('saved_plts_wts/14/mirl_results_roc.csv', index=False)
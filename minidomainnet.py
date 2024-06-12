import os.path as osp

import torch
import torch.nn.utils as utils
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob 
import random
from trainer.odgclip import *

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,num_classes)
  
    return image, domain, label, label_one_hot 


#################-------DATASET------#######################

domains = ['clipart', 'painting', 'sketch', 'real']

'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
path_dom1='./data/domainnet/'+domains[0]
domain_name1 = path_dom1.split('/')[-1]
dom1_classnames = []
dom1_filenames = []
with open('./data/domainnet/splits_mini/'+domains[0]+'_train.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0] 
            class_name = parts[-2]
            if class_name not in dom1_classnames:
                dom1_classnames.append(class_name)           
            filename_with_class_name = f"{class_name}/{filename}"
            dom1_filenames.append(filename_with_class_name)
dom1_classnames.sort()
class_names = dom1_classnames
num_classes = len(class_names)
c=0
index=0
index_dom1 = list(range(0,20)) + list(range(40,60))
for i in dom1_classnames:
    if index in index_dom1:
        paths = [filename for filename in dom1_filenames if filename.startswith(i + '/')]
        for j in paths:
            selected_paths = glob.glob(path_dom1 + '/' + j)
            random.shuffle(selected_paths)
            image_path_dom1.extend(selected_paths)
            label_class_dom1.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index+1
label_dom1=[0 for _ in range(len(image_path_dom1))]  


'''
############### The source dataset 2 ##################
'''

image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
path_dom2='./data/domainnet/'+domains[1]
domain_name2 = path_dom1.split('/')[-1]
dom2_classnames = []
dom2_filenames = []
with open('./data/domainnet/splits_mini/'+domains[1]+'_train.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0] 
            class_name = parts[-2]
            if class_name not in dom2_classnames:
                dom2_classnames.append(class_name)           
            filename_with_class_name = f"{class_name}/{filename}"
            dom2_filenames.append(filename_with_class_name)
dom2_classnames.sort()
class_names = dom2_classnames
num_classes = len(class_names)
c=0
index=0
index_dom2 = list(range(0,10)) + list(range(20,40)) + list(range(80,90))
for i in dom2_classnames:
    if index in index_dom2:
        paths = [filename for filename in dom2_filenames if filename.startswith(i + '/')]
        for j in paths:
            selected_paths = glob.glob(path_dom2 + '/' + j)
            random.shuffle(selected_paths)
            image_path_dom2.extend(selected_paths)
            label_class_dom2.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index+1
label_dom2=[1 for _ in range(len(image_path_dom2))] 


'''
############### The source dataset 3 ##################
'''

image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
path_dom3='./data/domainnet/'+domains[2]
domain_name3 = path_dom3.split('/')[-1]
dom3_classnames = []
dom3_filenames = []
with open('./data/domainnet/splits_mini/'+domains[2]+'_train.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0] 
            class_name = parts[-2]
            if class_name not in dom3_classnames:
                dom3_classnames.append(class_name)           
            filename_with_class_name = f"{class_name}/{filename}"
            dom3_filenames.append(filename_with_class_name)
dom3_classnames.sort()
class_names = dom1_classnames
num_classes = len(class_names)
c=0
index=0
index_dom3 = list(range(10,20)) + list(range(40,50)) + list(range(60,80))
for i in dom3_classnames:
    if index in index_dom3:
        paths = [filename for filename in dom3_filenames if filename.startswith(i + '/')]
        for j in paths:
            selected_paths = glob.glob(path_dom3 + '/' + j)
            random.shuffle(selected_paths)
            image_path_dom3.extend(selected_paths)
            label_class_dom3.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index+1
label_dom3=[2 for _ in range(len(image_path_dom3))]      

# Known Classes
index_dom = list(set(index_dom1 + index_dom2 + index_dom3))
known_class_names = [class_names[idx] for idx in index_dom]
known_classes = ",".join(known_class_names)
'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
domain_names=[]
domain_names.append(domain_name1)
domain_names.append(domain_name2)
domain_names.append(domain_name3)
print("domain_names",domain_names)
    
'''
##### Creating dataloader ######
'''
batchsize = 2
train_prev_ds=DataTrain(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')
train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

domain_prev = domain_prev.to(device)

class_names.sort()
train_prev_classnames = class_names[:90]


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def domain_text_loss(diff_textfeatures, domain):
    losses = []
    for i in range(len(domain) - 1):
        if domain[i] != domain[i + 1]:
           loss = F.mse_loss(diff_textfeatures[i], diff_textfeatures[i + 1])
           losses.append(loss)

    mse_loss = torch.mean(torch.stack(losses))

    return mse_loss


class ImageFilter(nn.Module):
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        grayscale_images = torch.mean(images, dim=1, keepdim=True)  # Convert to grayscale
        return grayscale_images.mean((2, 3))  # Calculate the average pixel value for each image

    def forward(self, image_tensor):
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)

        fraction_to_select = 1.0
        
        num_images_to_select = int(batch_size * fraction_to_select)
        indices_with_brightness_condition = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
        if len(indices_with_brightness_condition) < num_images_to_select:
           selected_indices = indices_with_brightness_condition
           num_black_images_to_select = num_images_to_select - len(indices_with_brightness_condition)
           all_indices = list(range(batch_size))
           black_indices = [i for i in all_indices if i not in indices_with_brightness_condition]
           random_black_indices = random.sample(black_indices, num_black_images_to_select)
           selected_indices += random_black_indices
           return selected_indices
        else:
           selected_indices = random.sample(indices_with_brightness_condition, num_images_to_select)
           return selected_indices

image_filter = ImageFilter(brightness_threshold=0.01)

def train_epoch(model, unknown_image_generator, domainnames, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img_prev, domain_prev, label_prev, label_one_hot_prev in tqdm_object:
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        batch = img_prev.shape[0]

        unknown_posprompt1 = domainnames[0].replace("_", " ") + " of an unknown object"
        generated_unknown_images1 = unknown_image_generator(batch, unknown_posprompt1, known_classes)

        unknown_posprompt2 = domainnames[1].replace("_", " ") + " of an unknown object"
        generated_unknown_images2 = unknown_image_generator(batch, unknown_posprompt2, known_classes)

        unknown_posprompt3 = domainnames[2].replace("_", " ") + " of an unknown object"
        generated_unknown_images3 = unknown_image_generator(batch, unknown_posprompt3, known_classes)

        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((len(domainnames)*generated_unknown_images1.shape[0],), unknown_label_rank).to(device)
        
        unknown_domain1 = torch.full((generated_unknown_images1.shape[0],), 0).to(device)
        unknown_domain2 = torch.full((generated_unknown_images2.shape[0],), 1).to(device)
        unknown_domain3 = torch.full((generated_unknown_images3.shape[0],), 2).to(device)


        generated_unknown_images = torch.cat((generated_unknown_images1, generated_unknown_images2, generated_unknown_images3), dim=0)
        unknown_domains = torch.cat((unknown_domain1, unknown_domain2, unknown_domain3), dim=0)
        random_indices = image_filter(generated_unknown_images) 
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domains[random_indices]
        
        img = torch.cat((img_prev, selected_images), dim=0)
        img = img.to(device)

        label = torch.cat((label_prev, selected_labels), dim=0)
        label = label.to(device)

        domain = torch.cat((domain_prev, selected_domains), dim=0)
        domain = domain.to(device)

        output, diff_projfeatures = model(img, label)
        output = output.to(device)
    
        crossentropy_loss = F.cross_entropy(output, label)

        text_loss = domain_text_loss(diff_projfeatures, domain)

        loss = crossentropy_loss + 0.01*(text_loss)
    
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)
        print("accuracy:", accuracy_meter.avg)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, accuracy=accuracy_meter.avg, lr=get_lr(optimizer))
    return loss_meter, accuracy_meter.avg
  
unknown_image_generator = GenerateUnknownImages().to(device)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')

train_model = CustomCLIP(train_classnames, domain_names, clip_model).to(device)

params = [
            {"params": train_model.domainclass_pl.parameters()},
            {"params": train_model.domain_pl.parameters()},
            {"params": train_model.conv_layer.parameters()},
            {"params": train_model.upsample_net.parameters()}
        ]

optimizer = torch.optim.AdamW(params,  weight_decay=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
scaler = GradScaler() 

'''
Test dataset
'''
test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_domain_names=[]
test_path_dom='./data/domainnet/'+domains[3]
test_domain_name = test_path_dom.split('/')[-1]
testdom_classnames = []
test_filenames = []
with open('./data/domainnet/splits_mini/real_test.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0]
            class_name = parts[-2]
            if class_name not in testdom_classnames:
                testdom_classnames.append(class_name)            
            filename_with_class_name = f"{class_name}/{filename}"
            test_filenames.append(filename_with_class_name)
testdom_classnames.sort()
test_classnames = []
print(len(test_filenames))
c=0
index=0
test_index = list(range(0,5))+list(range(8,18))+list(range(25,35))+list(range(43,48))+list(range(75,80))+list(range(83,88))+list(range(90,126))

for i in testdom_classnames:
    if index in test_index:
        paths = [filename for filename in test_filenames if filename.startswith(i + '/')]
        for j in paths:
            selected_paths = glob.glob(test_path_dom + '/' + j)
            random.shuffle(selected_paths)
            test_image_path_dom.extend(selected_paths)
            test_label_class_dom.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index+1
test_label_dom=[3 for _ in range(len(test_image_path_dom))]
  
test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)

test_label_class_final=[]
test_label_class_final_modified = [label if label <= 89 else 90 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)


test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)

'''
############### Making the test dataloader ##################
''' 

test_ds=DataTrain(test_image_path_final,test_label_dom_final,test_label_class_final)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

num_epochs = 20
step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = "./results/minidomainnet/real.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(train_model, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, step)
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")

    TRAIN_MODEL_PATH = Path("./train_models/minidomainnet/real")
    TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MODEL_NAME = f"real_{epoch+1}.pth"
    TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
    print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
    torch.save(obj=train_model.state_dict(), f=TRAIN_MODEL_SAVE_PATH)

    MODEL_PATH = "./train_models/minidomainnet/real"
    MODEL_NAME = f"real_{epoch+1}.pth"
    MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
    
    test_model = CustomCLIP(train_classnames, test_domain_names, clip_model).to(device)
    test_model.load_state_dict(torch.load(MODEL_FILE))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for test_img, test_domain, test_label, test_label_one_hot in test_tqdm_object:
            test_img = test_img.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            test_output, _ = test_model(test_img.to(device), test_label)

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 89) 
            class_b_mask = (test_label > 89)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()
        
        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        print(f"epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
        print(f"epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")

        average_acc = (2*closed_set_acc*open_set_acc)/(closed_set_acc + open_set_acc)
        print(f"epoch {epoch+1} : harmonic score: {average_acc}")

        accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Harmonic Score: {average_acc}%\n")
        accuracy_file.write("\n") 
        accuracy_file.flush()
        
        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path("./test_models/minidomainnet")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = "real.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()
        else:
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()

print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
print(f"Best harmonic score till now: {best_avg_acc}")

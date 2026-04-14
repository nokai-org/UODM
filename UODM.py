
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import time
from torchsummary import summary
import torchvision
import os
import numpy
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

N_CLASSES=5
IMG_SIZE=640
GRID=10
BATCH_SIZE=16
r_size=IMG_SIZE//GRID


def parse_labels(path):

    labels=torch.zeros((5, GRID, GRID))
    labels[1]+=0.5
    labels[2]+=0.5
    with open(path, 'r', encoding='utf-8') as file:

        lines = file.read().split("\n")

        for line in lines:

            s = line.split()
            s=list(map(lambda x: float(x), s))
            if len(s)<5:
              continue
            h=s[4]
            w=s[3]

            gx=IMG_SIZE*s[1]
            gy=IMG_SIZE*s[2]

            ix=int(gx//r_size)
            iy=int(gy//r_size)

            ix = min(ix, GRID-1)
            iy = min(iy, GRID-1)


            x=(gx%r_size)/r_size
            y=(gy%r_size)/r_size

            
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx = ix + dx
                    ny = iy + dy
            
                    if 0 <= nx < GRID and 0 <= ny < GRID:
                        dist_sq = dx**2 + dy**2  
                        sigma_sq = 0.8      
                        
                        p = math.exp(-dist_sq / (2 * sigma_sq))

                        
                        if p>labels[0][ny][nx] and ((abs(dx)==1 and abs(dy)==1) or (abs(dx)==0 and abs(dy)==1) or (abs(dx)==1 and abs(dy)==0)):
                            
                            sx=(dx*(-1)+1)/2
                            sy=(dy*(-1)+1)/2

                            labels[1][ny][nx]=sx
                            labels[2][ny][nx]=sy
                         
                        
                        labels[0][ny][nx] = max(labels[0][ny][nx], p)
    
            


            labels[0][iy][ix]=1


            labels[1][iy][ix]=x
            labels[2][iy][ix]=y
            labels[3][iy][ix]=w
            labels[4][iy][ix]=h

    return labels



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_images_from(data_dir):

    filenames = [name for name in sorted(os.listdir(data_dir))]

    batch_size = len(filenames)
    batch = torch.zeros(batch_size, 3, 640, 640, dtype=torch.uint8)

    for i, filename in enumerate(filenames):
        batch[i] = torchvision.io.read_image(os.path.join(data_dir, filename))

    batch = batch.float()
    batch /= 255.0

    return batch

def load_labels_from(data_dir):

    filenames = [name for name in sorted(os.listdir(data_dir))]

    batch_size = len(filenames)
    batch = torch.zeros(batch_size, 5, GRID, GRID, dtype=torch.float)


    for i, filename in enumerate(filenames):
        batch[i] = parse_labels(os.path.join(data_dir, filename))

    return batch


X=load_images_from(r"dataset\train\images")
y=load_labels_from(r"dataset\train\labels")

X_v=load_images_from(r"dataset\valid\images")
y_v=load_labels_from(r"dataset\valid\labels")

train_dataset = TensorDataset(X, y)
val_dataset = TensorDataset(X_v, y_v)


training_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

validation_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.signs = nn.Sequential(
            # 640 -> 320
            nn.Conv2d(3, 32, 3, padding=1,stride=2), 
            nn.BatchNorm2d(32), 
            nn.SiLU(), 
        
            nn.Conv2d(32, 64, 3, padding=1,stride=2), 
            nn.BatchNorm2d(64), 
            nn.SiLU(), 
            
        
            nn.Conv2d(64, 128, 3, padding=1,stride=2), 
            nn.BatchNorm2d(128), 
            nn.SiLU(), 
            
        
            nn.Conv2d(128, 128, 3, padding=1,stride=2), 
            nn.BatchNorm2d(128), 
            nn.SiLU(), 
            
        
            nn.Conv2d(128, 128, 3, padding=1,stride=2), 
            nn.BatchNorm2d(128), 
            nn.SiLU(), 
            
        
            nn.Conv2d(128, 144, 3, padding=1,stride=2), 
            nn.BatchNorm2d(144), 
            nn.SiLU(), 
        
            nn.Conv2d(144, 192, 3, padding=1), 
            nn.BatchNorm2d(192), 
            nn.SiLU(), 
            nn.Conv2d(192, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.SiLU(), 
        )

        self.cl = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 5, kernel_size = 1),
        )

    def forward(self, x):
        x=self.signs(x)
        x=self.cl(x)
        return x






model=CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

MSE=torch.nn.MSELoss()
sg = nn.Sigmoid()

FocalLoss=torchvision.ops.focal_loss.sigmoid_focal_loss


def custom_loss(outputs, labels):
    global o,b,c,b1
    
    mask=labels[:,0,:,:]==1
    
    mask2=labels[:,0,:,:]>0
    
  

    loss=FocalLoss(((outputs[:,0,:,:])),((labels[:,0,:,:])))
    
    
    
    
    loss0 = MSE(sg(outputs[:,1,:,:][mask2]),labels[:,1,:,:][mask2])
    loss1 = MSE(sg(outputs[:,2,:,:][mask2]),labels[:,2,:,:][mask2])
    
    loss2 = MSE(sg(outputs[:,3,:,:][mask]),labels[:,3,:,:][mask])
    loss3 = MSE(sg(outputs[:,4,:,:][mask]),labels[:,4,:,:][mask])

    b_loss=loss0+loss1+1*(loss2+loss3)

 
    
    return b_loss






def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    idx=0
    global lbl,out
    for i, data in enumerate(training_loader):
    

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

      

        loss = custom_loss(outputs, labels)
        loss.backward()

        optimizer.step()

        idx+=1
        last_loss += loss.item()



    return last_loss/idx



timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 30

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number)


    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs=vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = custom_loss(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

   





    epoch_number += 1



torch.save(model.state_dict(), r"savedir")


model.load_state_dict(torch.load(r"savedir", weights_only=True))

def load_image_from(data_dir):


    batch = torch.zeros(1, 3, 640, 640, dtype=torch.uint8)

    batch[0] = torchvision.io.read_image(data_dir)

    batch = batch.float()
    batch /= 255.0

    return batch

sm = nn.Softmax()
exp = torch.exp

# TEST OF MODEL OUTPUTS

def clamp(n, _min, _max):  
    if n < _min:  
        return _min  
    elif n > _max:  
        return _max  
    else:  
        return n 

def xywh_to_xyxy(boxes):
    x, y, w, h = boxes
    
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    x1=clamp(x1,0,1)
    y1=clamp(y1,0,1)
    x2=clamp(x2,0,1)
    y2=clamp(y2,0,1)
    
    return [x1, y1, x2, y2]
    
def test(x):
    
    sm=torch.nn.Softmax()
    sg=torch.nn.Sigmoid()
    
    fig, ax = plt.subplots()
    
    
    inpt=load_image_from(r"imgdir")
  
    
    inpt=inpt.to(device)
    
    
    model.eval()
    out=model(inpt)
    inpt=inpt[0]
    out=out[0]

    
    
    

    
    out[0]= sg(out[0])  
    out[1]= sg(out[1])  
    out[2]= sg(out[2])  
    out[3]= sg(out[3])  
    out[4]= sg(out[4])  


    
    
    threshold=0.25
    
    
    bboxes=[]
    bboxes_xywh=[]
    scores=[]
    for i in range(GRID):
        for j in range(GRID):
            if out[0][i][j]>=threshold:
           
                x=int(out[1][i][j]*r_size)+j*r_size
                y=int(out[2][i][j]*r_size)+i*r_size
                w=int(out[3][i][j]*IMG_SIZE)
                h=int(out[4][i][j]*IMG_SIZE)
                
             
                

                
                #x=int(0.5*r_size)+j*r_size
                #y=int(0.5*r_size)+i*r_size
                #w=int(r_size)
                #h=int(r_size)
                
            
                x-=w//2
                y-=h//2
                
                x/=IMG_SIZE
                y/=IMG_SIZE
                w/=IMG_SIZE
                h/=IMG_SIZE
                

                
                bboxes_xywh.append([x,y,w,h])
                bbox=xywh_to_xyxy([x,y,w,h])
                bboxes.append(bbox)
                scores.append(out[0][i][j].item())
                
                #bboxes.append(["",float(out[0][i][j]),x,y,w,h])

    
    scores=torch.tensor(scores)
    bboxes=torch.tensor(bboxes)

    indices=torchvision.ops.nms(bboxes,scores,0.15)
    
    
    inpt=inpt.permute(1,2,0)
    
    ax.imshow(inpt.cpu())
    
    for i in indices:
        x,y,w,h=bboxes_xywh[i]

        x*=IMG_SIZE
        y*=IMG_SIZE
        w*=IMG_SIZE
        h*=IMG_SIZE
        
        rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        rect2 = matplotlib.patches.Rectangle((x, y-30), w, 30, linewidth=2, edgecolor='black', facecolor='black')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        
        ax.text(x,y-10,str(round(scores[i].item()*100)/100),color="white",
             fontdict={"fontsize": 10,"fontweight":'bold',"ha":"left","va":"center"})
    '''
    for el in bboxes:
           
        rect = matplotlib.patches.Rectangle((el[2], el[3]), el[4], el[5], linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        ax.text(el[2]+10,el[3]+30,el[0]+": "+str( round(el[1]*100)/100),color="black",
             fontdict={"fontsize": 12,"fontweight":'normal',"ha":"left","va":"center"})
    '''
    
    plt.show()

test("")

#LABEL TEST

def label_test():
    sg=torch.nn.Sigmoid()
    
    fig, ax = plt.subplots()

    
 
    #inpt=load_image_from(r"D:\govnoXai\yolo_test\cndy640-splitted\valid\images\a094bc6e-candy_135.jpg")
    inpt=load_image_from(r"D:\govnoXai\yolo_test\cndy640-splitted\valid\images\d2179503-candy_32.jpg")
    
    
    
    inpt=inpt.to(device)
    
    inpt=inpt[0]



    #out=parse_labels(r"D:\govnoXai\yolo_test\cndy640-splitted\valid\labels\a094bc6e-candy_135.txt")
    out=parse_labels(r"D:\govnoXai\yolo_test\cndy640-splitted\valid\labels\d2179503-candy_32.txt")
    
    
    threshold=0.3
    
    
    bboxes=[]
    bboxes_xywh=[]
    scores=[]
    for i in range(GRID):
        for j in range(GRID):
            if out[0][i][j]>=threshold:
           
                x=int(out[1][i][j]*r_size)+j*r_size
                y=int(out[2][i][j]*r_size)+i*r_size
                w=int(out[3][i][j]*IMG_SIZE)
                h=int(out[4][i][j]*IMG_SIZE)

                
               
                
                #x=int(0.5*r_size)+j*r_size
                #y=int(0.5*r_size)+i*r_size
                #w=int(r_size)
                #h=int(r_size)
                
                x-=w//2
                y-=h//2
            
                bboxes.append(["",float(out[0][i][j]),x,y,w,h])

   
    inpt=inpt.permute(1,2,0)
    
    ax.imshow(inpt.cpu())

    for el in bboxes:
           
        rect = matplotlib.patches.Rectangle((el[2], el[3]), el[4], el[5], linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        #ax.text(el[2]+10,el[3]-150,el[0]+": "+str( round(el[1]*100)/100),color="black",
         #    fontdict={"fontsize": 12,"fontweight":'normal',"ha":"left","va":"center"})
    
    
    plt.show()

label_test()

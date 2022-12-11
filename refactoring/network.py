import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import time
import numpy as np
import torchvision as tv
def get_num_correct(preds, labels, length, device): 
    cont = 0
    index = torch.argmax(preds)
    preds = [0] * length
    preds[index] = 1
    preds = torch.Tensor(preds)
    preds = preds.to(device)
    for i, a_ in enumerate(preds):
        if (a_ == labels[i] and labels[i] == 1) or (a_ == labels[i] and labels[i] == 0):
            cont += 1
        else:
            cont -= 1        
    return cont

turno = True
class Network(nn.Module):
    def __init__(self, faseFinale):
        super(Network, self).__init__()
        self.faseFinale = faseFinale
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, stride=2, kernel_size=7, padding = 4),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=7, padding = 'same'),
            nn.BatchNorm2d(12))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels= 25, kernel_size=3, padding = 3), 
            nn.BatchNorm2d(25))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels= 75, kernel_size=3, padding = 2), 
            nn.BatchNorm2d(75))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels= 150, kernel_size=3, padding = 2),
            nn.Conv2d(in_channels=150, out_channels= 150, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(150))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=150, out_channels= 350, kernel_size=3, padding = 2), 
            nn.BatchNorm2d(350))

        self.conv6 = nn.Conv2d(in_channels=350, out_channels= 350, kernel_size=3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels=350, out_channels= 1024, kernel_size=3)

        self.relu = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(0.05)
        self.drop2d = nn.Dropout2d(0.05)

        #Global label
        self.class_fc0 = nn.Linear(in_features=1024, out_features=750)
        self.class_fc0_1 = nn.Linear(in_features=750, out_features=512)
        self.class_fc0_2 = nn.Linear(in_features=512, out_features=300)

        #Global box
        self.box_fc0 = nn.Linear(in_features=1024, out_features=750)
        self.box_fc0_1 = nn.Linear(in_features=750, out_features=512)

        #first label
        self.class_fc1_0 = nn.Linear(in_features=300, out_features=150)
        self.class_fc1_1 = nn.Linear(in_features=150, out_features=20)
        self.class_out1 = nn.Linear(in_features=20, out_features=7)

        #first box
        self.box_fc1_0 = nn.Linear(in_features=512, out_features=300)
        self.box_fc1_1 = nn.Linear(in_features=300, out_features=150)
        self.box_fc1_2 = nn.Linear(in_features=150, out_features=20)
        self.box_out1 = nn.Linear(in_features=20, out_features=4)

        #second label
        self.class_fc2_0 = nn.Linear(in_features=300, out_features=150)
        self.class_fc2_1 = nn.Linear(in_features=150, out_features=20)
        self.class_out2 = nn.Linear(in_features=20, out_features=6)

        #Second box
        self.box_fc2_0 = nn.Linear(in_features=512, out_features=300)
        self.box_fc2_1 = nn.Linear(in_features=300, out_features=150)
        self.box_fc2_2 = nn.Linear(in_features=150, out_features=20)
        self.box_out2 = nn.Linear(in_features=20, out_features=4)      
        

    def forward(self, t):
        global turno
        t = self.conv1(t)
        t = self.relu(t)
        t = self.max_pool(t)
        t = self.drop2d(t)

        t = self.conv2(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv3(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv4(t)
        t = self.relu(t)
        t = self.max_pool(t)
        t = self.drop2d(t)

        t = self.conv5(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv6(t) 

        t = self.conv7(t) 

        t = torch.flatten(t, start_dim=1)

        #global label
        class_t = self.class_fc0(t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc0_1(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc0_2(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        #first label
        class_t1 = self.class_fc1_0(class_t)
        class_t1 = self.relu(class_t1)
        class_t1 = self.drop(class_t1)

        class_t1 = self.class_fc1_1(class_t1)
        class_t1 = self.relu(class_t1)
        class_t1 = self.drop(class_t1)

        class_t1 = self.class_out1(class_t1)

        class_t1 = F.softmax(class_t1, dim=1)

        #second label
        class_t2 = self.class_fc2_0(class_t)
        class_t2 = self.relu(class_t2)
        class_t2 = self.drop(class_t2)

        class_t2 = self.class_fc2_1(class_t2)
        class_t2 = self.relu(class_t2)
        class_t2 = self.drop(class_t2)

        class_t2 = self.class_out2(class_t2)

        class_t2 = F.softmax(class_t2, dim=1)

        #global box
        box_t = self.box_fc0(t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)
        
        box_t = self.box_fc0_1(box_t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)

        #First box
        box_t1 = self.box_fc1_0(box_t)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_fc1_1(box_t1)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_fc1_2(box_t1)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_out1(box_t1)
        box_t1 = F.sigmoid(box_t1)

        #Second box
        box_t2 = self.box_fc2_0(box_t)
        box_t2 = self.relu(box_t2)
        box_t2 = self.drop(box_t2)

        box_t2 = self.box_fc2_1(box_t2)
        box_t2 = self.relu(box_t2)
        box_t2 = self.drop(box_t2)

        box_t2 = self.box_fc2_2(box_t2)
        box_t2 = self.relu(box_t2)
        box_t2 = self.drop(box_t2)

        box_t2 = self.box_out2(box_t2)
        box_t2 = F.sigmoid(box_t2)

        return [class_t1, box_t1, class_t2, box_t2]

def initialize_weights(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

def train(num_of_epochs, lr, dataset, dataset2, valdataset, samples, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=78, shuffle=True)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=78, shuffle=True)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=78, shuffle=True)

    model = Network(False)
    model.apply(initialize_weights)
    model = model.to(device)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    epochs = []
    losses = []
    turno = True
    primaVolta = True
    primaVolta2 = True
    # Creating a directory for storing models
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    for epoch in range(num_of_epochs):
        tot_loss = 0
        tot_correct = 0
        train_start = time.time()
        model.train()


        if turno:
            if primaVolta:
                print("Metto a zero quelli che non mi interessano")
                c = 0
                for child in model.children():
                    c += 1
                    if c >= 24:
                        for param in child.parameters():
                            param.requires_grad = False
                    else:
                        for param in child.parameters():
                            param.requires_grad = True
                primaVolta = False

            for batch, (x, y, z) in enumerate(dataloader):
                
                x,y,z = x.to(device),y.to(device),z.to(device)
                class_loss = 0
                optimizer.zero_grad()
                [y_pred, z_pred, y_pred1, z_pred1]= model(x)
                box_loss = 0 
                
                for i, yR in enumerate(y): 
                    array1 = [0] * 7
                    array1[yR[0]] = 1
                    array1 = torch.FloatTensor(array1)
                    array1 = array1.to(device)
                    class_loss += F.cross_entropy(y_pred[i], array1) 
                    #class_loss += F.cross_entropy(y_pred1[i], array1)

                    #box_loss += tv.ops.distance_box_iou_loss(z_pred[i], z[i])
                    #box_loss += tv.ops.distance_box_iou_loss(z_pred1[i], z[i])
                    box_loss += F.mse_loss(z_pred[i], z[i])

                (class_loss+box_loss).backward()
                optimizer.step()

                print("Train batch:", batch+1, " epoch: ", epoch, " ", (time.time()-train_start)/60, end='\r')
            if epoch%20 == 0 and (not epoch == 0):
                turno = False
                primaVolta = True
                primaVolta2 = True

        elif not turno:
            
            if primaVolta2:

                print("Metto a True e poi a False quelli che non mi interessano")
                c = 0
                for child in model.children():
                    c += 1
                    if c >= 17 and c <= 23:
                        for param in child.parameters():
                            param.requires_grad = False
                    else:
                        for param in child.parameters():
                            param.requires_grad = True
                primaVolta2 = False

            for batch, (x, y, z) in enumerate(dataloader2):
                
                x,y,z = x.to(device),y.to(device),z.to(device)
                class_loss = 0
                optimizer.zero_grad()
                [y_pred, z_pred, y_pred1, z_pred1]= model(x)
                box_loss = 0 
                
                for i, yR in enumerate(y): 
                    array1 = [0] * 6
                    array1[yR[0]-7] = 1
                    array1 = torch.FloatTensor(array1)
                    array1 = array1.to(device)
                    #class_loss += F.cross_entropy(y_pred[i], array1) 
                    class_loss += F.cross_entropy(y_pred1[i], array1)

                    #box_loss += tv.ops.distance_box_iou_loss(z_pred[i], z[i])
                    #box_loss += tv.ops.distance_box_iou_loss(z_pred1[i], z[i])
                    box_loss += F.mse_loss(z_pred1[i], z[i])

                (class_loss+box_loss).backward()
                optimizer.step()

                print("Train batch:", batch+51, " epoch: ", epoch, " ", (time.time()-train_start)/60, end='\r')      
            if epoch%20 == 0:
                turno = True
                primaVolta2 = True
                primaVolta = True
        model.eval()

        for batch, (x, y, z) in enumerate(valdataloader):
            x,y,z = x.to(device),y.to(device),z.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                x,y,z = x.to(device),y.to(device),z.to(device)
                class_loss = 0
                optimizer.zero_grad()
                [y_pred, z_pred, y_pred1, z_pred1]= model(x)
                box_loss = 0 
                for i, yR in enumerate(y): 
                    if yR < 7:
                        array1 = [0] * 7
                        array1[yR] = 1
                        array1 = torch.FloatTensor(array1)
                        array1 = array1.to(device)
                        class_loss += F.cross_entropy(y_pred[i], array1) 
                        tot_correct += get_num_correct(y_pred[i], array1, 7, device)
                    else:
                        array2 = [0] * 6
                        array2[yR-7] = 1
                        array2 = torch.FloatTensor(array2)
                        array2 = array2.to(device)
                        class_loss += F.cross_entropy(y_pred1[i], array2) 
                        tot_correct += get_num_correct(y_pred1[i], array2, 6, device)

                    box_loss += F.mse_loss(z_pred[i], z[i])
                    box_loss += F.mse_loss(z_pred1[i], z[i])

            tot_loss += (class_loss.item() + box_loss.item())  #.item()
            

            print("Test batch:", batch+1, " epoch: ", epoch, " ", (time.time()-train_start)/60, end='\r')

        epochs.append(epoch)
        losses.append(tot_loss)

        print("Epoch", epoch, "Accuracy: ", tot_correct/int(6.5*1864), "loss: ", tot_loss/1864, 
        " time: ", (time.time()-train_start)/60, " mins")
        if epoch%5 == 0:
            torch.save(model.state_dict(), savedir+"/model_ep"+str(epoch+1)+".pth")
    
    #print(losses)
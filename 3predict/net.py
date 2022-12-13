from torchviz import make_dot   
import torch 
import torch.nn as nn
import torch.nn.functional as F


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
        self.class_out1 = nn.Linear(in_features=20, out_features=5)

        #first box
        self.box_fc1_0 = nn.Linear(in_features=512, out_features=300)
        self.box_fc1_1 = nn.Linear(in_features=300, out_features=150)
        self.box_fc1_2 = nn.Linear(in_features=150, out_features=20)
        self.box_out1 = nn.Linear(in_features=20, out_features=4)

        #second label
        self.class_fc2_0 = nn.Linear(in_features=300, out_features=150)
        self.class_fc2_1 = nn.Linear(in_features=150, out_features=20)
        self.class_out2 = nn.Linear(in_features=20, out_features=4)

        #Second box
        self.box_fc2_0 = nn.Linear(in_features=512, out_features=300)
        self.box_fc2_1 = nn.Linear(in_features=300, out_features=150)
        self.box_fc2_2 = nn.Linear(in_features=150, out_features=20)
        self.box_out2 = nn.Linear(in_features=20, out_features=4)      
        
        #third label
        self.class_fc3_0 = nn.Linear(in_features=300, out_features=150)
        self.class_fc3_1 = nn.Linear(in_features=150, out_features=20)
        self.class_out3 = nn.Linear(in_features=20, out_features=4)

        #third box
        self.box_fc3_0 = nn.Linear(in_features=512, out_features=300)
        self.box_fc3_1 = nn.Linear(in_features=300, out_features=150)
        self.box_fc3_2 = nn.Linear(in_features=150, out_features=20)
        self.box_out3 = nn.Linear(in_features=20, out_features=4)      

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

        #third label
        class_t3 = self.class_fc3_0(class_t)
        class_t3 = self.relu(class_t3)
        class_t3 = self.drop(class_t3)

        class_t3 = self.class_fc3_1(class_t3)
        class_t3 = self.relu(class_t3)
        class_t3 = self.drop(class_t3)

        class_t3 = self.class_out3(class_t3)

        class_t3 = F.softmax(class_t3, dim=1)

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

        #Third box
        box_t3 = self.box_fc3_0(box_t)
        box_t3 = self.relu(box_t3)
        box_t3 = self.drop(box_t3)

        box_t3 = self.box_fc3_1(box_t3)
        box_t3 = self.relu(box_t3)
        box_t3 = self.drop(box_t3)

        box_t3 = self.box_fc3_2(box_t3)
        box_t3 = self.relu(box_t3)
        box_t3 = self.drop(box_t3)

        box_t3 = self.box_out3(box_t3)
        box_t3 = F.sigmoid(box_t3)

        return [class_t1, box_t1, class_t2, box_t2, class_t3, box_t3]


def print_model(dataloader):
    model = Network(False)
    model = model.to('cpu')

    for (x, y, z) in dataloader:
        x1 = x
        break
    yhat = model(x1)

    tupla = tuple(yhat)
    make_dot(tupla, params=dict(list(model.named_parameters())), show_attrs=False).render("cnn_torchviz", format="png")
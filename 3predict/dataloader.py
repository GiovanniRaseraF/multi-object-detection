import os
import cv2
import json
import torch
import numpy as np

def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    return ret

def process_dataset(image_dir = "./assignment_1/train/images", ann_dir = "./assignment_1/train/annotations/", samples = 200, scale = 112, active_train = False, changes={}):
    if not active_train: 
        return [], [], []
    
    # ret values
    labels  = []
    boxes   = []
    images  = []

    # loading
    samples = int(samples)
    rows = os.listdir(image_dir)[:samples]
    i = 0
    for row in rows[:samples]:
        imageinfo_filename = str(row[int(0):int(6)]) + ".json"

        with open(ann_dir + imageinfo_filename, 'r') as f:
            data = json.loads(f.read())
            array = items(data)

            # loading and scaling 
            img_path = row
            image = cv2.imread(os.path.join(image_dir + "/",img_path))
            h = image.shape[0] / scale
            w = image.shape[1] / scale
            image = cv2.resize(image, (scale, scale))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / (255.0) 
            #labels.append([])
            #boxes.append([])
             
            for item in array:   #migliorare
                lista = item['bounding_box']
                lista[0] = lista[0]/scale/w
                lista[1] = lista[1]/scale/h
                lista[2] = lista[2]/scale/w
                lista[3] = lista[3]/scale/h

                boxes.append(lista)
                labels.append(item['category_name'])
                images.append(image)
           
            i += 1
    t_labels = []
    t_boxes = []

    #for label in labels:
        #t_labels.append([changes[label[i]] if i<len(label) else -1 for i in range(max(6, len(label)))])

   #for box in boxes:
        #t_boxes.append([box[i] if i<len(box) else [-1, -1, -1, -1] for i in range(max(6, len(box)))])

    for label in labels:
        t_labels.append([changes[label]])
    
    t_images = np.array(images)
    t_labels = np.array(t_labels)
    t_boxes = np.array(boxes)

    return t_images, t_labels, t_boxes


class Dataset():
    def __init__(self, train_images, train_labels, train_boxes):
        self.images = torch.permute(torch.from_numpy(train_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(train_boxes).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx],
              self.labels[idx],
              self.boxes[idx])

class ValDataset(Dataset):

    def __init__(self, val_images, val_labels, val_boxes):
        self.images = torch.permute(torch.from_numpy(val_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(val_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(val_boxes).float()
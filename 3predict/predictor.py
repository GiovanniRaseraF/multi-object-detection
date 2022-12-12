from network import Network
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torch.optim as optim
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    return ret

def preprocess(img, image_size):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image

def postprocess(image, results, classes, scale):
    [class_probs, bounding_box, class_probs1, bounding_box1, class_probs2, bounding_box2] = results
    class_index = torch.argmax(class_probs)
    class_label = classes[class_index]

    class_index1 = torch.argmax(class_probs1) + 5
    class_label1 = classes[class_index1]

    class_index2 = torch.argmax(class_probs2) + 9
    class_label2 = classes[class_index2]
    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]
    x3, y3, x4, y4 = bounding_box1[0]
    x5, y5, x6, y6 = bounding_box2[0]

    # # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(scale * x1)
    x2 = int(scale * x2)
    y1 = int(scale * y1)
    y2 = int(scale * y2)

    x3 = int(scale * x3)
    x4 = int(scale * x4)
    y3 = int(scale * y3)
    y4 = int(scale * y4)

    x5 = int(scale * x5)
    x6 = int(scale * x6)
    y5 = int(scale * y5)
    y6 = int(scale * y6)
    # return the lable and coordinates

    return class_label, (x1,y1,x2,y2), torch.max(class_probs)*100, class_label1, (x3,y3,x4,y4), torch.max(class_probs1)*100, class_label2, (x5,y5,x6,y6), torch.max(class_probs2)*100, 

def predict(image, number_model, scale, showfinalimage, saveimage, classes, savedir, recall = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network(True)
    model = model.to(device)
    #model.load_state_dict(torch.load(savedir+"/model_ep"+str(number_model)+".pth"))
    model.load_state_dict(torch.load(savedir+"/model_ep"+str(number_model)+".pth", map_location=torch.device('cpu')))
    model.eval()

    # Reading Image
    img  = cv2.imread(image)

    # # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(img, scale)
    result = model(torch.permute(torch.from_numpy(processed_image).float(),(0,3,1,2)).to(device))
    scale = 448
    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence, label1, (x3, y3, x4, y4), confidence1, label2, (x5, y5, x6, y6), confidence2 = postprocess(image, result, classes, scale)
    
    img = cv2.resize(img, (scale, scale))
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
    img = cv2.rectangle(img, (x3, y3), (x4, y4), (255, 153, 51), 2)
    img = cv2.rectangle(img, (x5, y5), (x6, y6), (0, 153, 51), 2)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    cv2.putText(img, label1, (x4-50, y4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 51), 2)
    cv2.putText(img, label2, (x6-50, y6+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 51), 2)
    

    #print("Informazioni")
    
    labels = [label, label1, label2]
    confidences = [confidence.item(), confidence1.item(), confidence2.item()]
    for l, c in zip(labels, confidences):
        if c >= 70:
            pass #print(l, c)

    imageinfo_filename = str(image[-10:-4]) + ".json"
    
    with open("./assignment_1/test/all_annotations/" + imageinfo_filename, 'r') as f:
        data = json.loads(f.read())
        array = items(data)
    if recall:
        print(image[-10:], len(array))  
        for item in array:
            print(f"{item['category_name']}, {label}, {label1}, {label2}, {item['category_name'] in labels}")

            
    
    # Showing and saving predicted
    plt.imshow(img[:,:,::-1])
    if saveimage:
        plt.savefig("./immagini/" + image[-10:])
    if showfinalimage:
        plt.show()
    #qui dobbiamo fare un bel return con quante ne predice giuste
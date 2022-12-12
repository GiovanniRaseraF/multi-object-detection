import os
import dataloader 
from dataloader import Dataset
from dataloader import ValDataset
from dataloader import process_dataset as dataset
from network import *
import sys
from predictor import *

# Settable parameters
#Get-Content input.txt |  C:/Users/fdsa/anaconda3/python.exe main.py 112 18000 1 2001 0 2001 0.000001 . ./models2

scale           = 112 
samples         = 9000
samples         = 9000
active_train    = False
num_of_epochs   = 2001
train_image_dir = '/assignment_1/train/images'
train_ann_dir = '/assignment_1/train/annotations'
val_image_dir   = '/assignment_1/test/images'
load_model      = not active_train 
number_model    = 2001
lr              = 0.000001

# specigic
classes = [
    'long sleeve dress',
    'long sleeve outwear',
    'long sleeve top', 
    'short sleeve dress', 
    'short sleeve outwear', 
    'short sleeve top', 
    'shorts', 
    'skirt', 
    'sling',
    'sling dress',
    'trousers', 
    'vest', 
    'vest dress'
]

changes = {
	"long sleeve dress"     : 0,
    "long sleeve outwear"   : 1,
    "long sleeve top"       : 2,
    "short sleeve dress"    : 3,
    "short sleeve outwear"  : 4,
    "short sleeve top"      : 5,
    "shorts"                : 6,
    "skirt"                 : 7,
    "sling"                 : 8,
    "sling dress"           : 9,
    "trousers"              : 10,
    "vest"                  : 11,
    "vest dress"            : 12
}

def printdict(dictionary):
    for key in (dictionary.items()):
        print(f"{key}")

test_samples = 0
accuracy = 0

if __name__ == "__main__":
    print("usage: main.py scale samples samples2 active_train num_of_epochs load_model number_model lr from")
    infodict = {
        "scale"           : int(sys.argv[1]),
        "samples"         : int(sys.argv[2]),
        "samples2"        : int(sys.argv[3]),
        "samples3"        : int(sys.argv[4]),
        "active_train"    : bool(int(sys.argv[5])),
        "num_of_epochs"   : int(sys.argv[6]),
        "load_model"      : bool(int(sys.argv[7])),
        "number_model"    : int(sys.argv[8]),
        "lr"              : float(sys.argv[9]),
        "fromdir"         : sys.argv[10].strip(),
        "savedir"         : sys.argv[11].strip(),
    }
    scale           = infodict["scale"] 
    samples         = infodict["samples"]
    samples2        = infodict["samples2"]
    samples3        = infodict["samples3"]
    active_train    = infodict["active_train"]
    num_of_epochs   = infodict["num_of_epochs"]
    load_model      = infodict["load_model"] 
    number_model    = infodict["number_model"] 
    lr              = infodict["lr"] 
    fromdir         = infodict["fromdir"]
    savedir         = infodict["savedir"]
    printdict(dictionary=infodict)

    #Get-Content input.txt | C:/Users/fdsa/anaconda3/python.exe main.py 112 9359 9000 0 2001 1 461 0.00001 . ./models
    
    # Loading training dataset
    train_images, train_labels, train_boxes = dataset(
        image_dir   = fromdir + train_image_dir,
        ann_dir     = fromdir + train_ann_dir +'/',
        samples     = samples,
        scale       = scale,
        active_train= active_train,
        changes     = changes
    ) 

    train_images2, train_labels2, train_boxes2 = dataset(
        image_dir   = fromdir + train_image_dir + '2',
        ann_dir     = fromdir + train_ann_dir + '2/',
        samples     = samples2,
        scale       = scale,
        active_train= active_train,
        changes     = changes
    )

    train_images3, train_labels3, train_boxes3 = dataset(
        image_dir   = fromdir + train_image_dir + '3',
        ann_dir     = fromdir + train_ann_dir + '3/',
        samples     = samples3,
        scale       = scale,
        active_train= active_train,
        changes     = changes
    )


    # Loading test dataset
    val_images, val_labels, val_boxes = dataset(
        image_dir   = fromdir + val_image_dir,
        ann_dir     = fromdir + "/assignment_1/test/annotations/",
        samples     = 4,
        scale       = scale,
        active_train= active_train,
        changes     = changes
    ) 

    

    # Train model
    if active_train:
        datas = Dataset(train_images, train_labels, train_boxes)
        datas2 = Dataset(train_images2, train_labels2, train_boxes2)
        datas3 = Dataset(train_images3, train_labels3, train_boxes3)
        valdataset = ValDataset(val_images, val_labels, val_boxes)

        train(
            num_of_epochs   = num_of_epochs,
            lr              = lr,
            dataset         = datas,
            dataset2        = datas2,
            dataset3        = datas3,
            valdataset      = valdataset,
            samples         = samples,
            savedir         = savedir
        )
    
    # Predict    
    while(True):
        imagename = input("Image:> ")
        if imagename == "end":
            exit(0)
        predict(
            image          = fromdir + val_image_dir + "/" + imagename + ".jpg", 
            number_model   = number_model,
            scale          = scale,
            showfinalimage = False,
            saveimage      = False,
            classes        = classes,
            savedir        = savedir,
            recall         = True
        )

   #L'accuracy è 0.3124048706240487 su 1095 foto

   #preso il return del predict, su 3000 foto vediamo quante hanno predetto giusto e aggiustiamo
   #la soglia di confidenza, poi in seguito calcoliamo recall e tutto il resto con il quale
   #creiamo la tabellina. Per LA intersection Union invece vediamo.  
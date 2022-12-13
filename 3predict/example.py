predicted = {
	"long sleeve dress"     :0,
    "long sleeve outwear"   :0,
    "long sleeve top"       :0,
    "short sleeve dress"    :0,
    "short sleeve outwear"  :0,
    "short sleeve top"      :0,
    "shorts"                :0,
    "skirt"                 :0,
    "sling"                 :0,
    "sling dress"           :0,
    "trousers"              : 0,
    "vest"                  : 0,
    "vest dress"            :0,
}

category = {
   "long sleeve dress"     :0,
    "long sleeve outwear"   :0,
    "long sleeve top"       :0,
    "short sleeve dress"    :0,
    "short sleeve outwear"  :0,
    "short sleeve top"      :1,
    "shorts"                :1,
    "skirt"                 :1,
    "sling"                 :1,
    "sling dress"           :2,
    "trousers"              : 2,
    "vest"                  : 2,
    "vest dress"            :2,
}

real = {
	"long sleeve dress"     :predicted.copy(),
    "long sleeve outwear"   :predicted.copy(),
    "long sleeve top"       :predicted.copy(),
    "short sleeve dress"    :predicted.copy(),
    "short sleeve outwear"  :predicted.copy(),
    "short sleeve top"      :predicted.copy(),
    "shorts"                :predicted.copy(),
    "skirt"                 :predicted.copy(),
    "sling"                 :predicted.copy(),
    "sling dress"           :predicted.copy(),
    "trousers"              : predicted.copy(),
    "vest"                  : predicted.copy(),
    "vest dress"            :predicted.copy(),
}

def intersection_over_union(bo_a, bo_b):
    box_a = []; box_b = []
    bo_a = bo_a[1:-1].split(", "); 
    bo_b = bo_b[1:-1].split(", ")
    for a,b in zip(bo_a, bo_b):
        if a == '':
            box_a.append(0)
        else:
            box_a.append(int(a))
        if b == '':
            box_b.append(0)
        else:
            box_b.append(int(b))
    width_a   =  box_a[2] - box_a[0]
    height_a  =  box_a[3] - box_a[1]
    left_a    =  box_a[0]
    top_a     =  box_a[1]

    width_b   =  box_b[2] - box_b[0]
    height_b  =  box_b[3] - box_b[1]
    left_b    =  box_b[0]
    top_b     =  box_b[1]

    xA = max(left_a, left_b)
    yA = max(top_a, top_a)
    xB = min(left_a+width_a, left_b+width_b)
    yB = min(top_a+height_a, top_b+height_b)

    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    box_a_area = (width_a + 1) * (height_a + 1)
    box_b_area = (width_b + 1) * (height_b + 1)

    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection + 0.0000001)

    return iou

if __name__ == "__main__":
    filein = open("pred.txt", "r")
    bb_accuracy = 0
    while(filein):
        line = filein.readline()
        if line == "end": break

        vals = line.split(" ")
        
        nums = vals[2]
        for i in range(int(nums)):
            readpred = filein.readline().split(";")

            realval = readpred[0].strip()
            realBB = readpred[5].strip()
            items = [readpred[1].strip(), readpred[2].strip(), readpred[3].strip()]
            bb = [readpred[6].strip(), readpred[7].strip(), readpred[8].strip()]
            position = category[realval]

            if realval in items:
                real[realval][realval] += 1
                predBB = bb[position] 
                bb_accuracy += intersection_over_union(realBB, predBB)
            else:
                wrongprediction = items[position]
                real[realval][wrongprediction] += 1

    '''for key, val in real.items():
        print(f"{key:30}", end = " ")
        for kk, vv in val.items():
            print(f"{vv:5}", end = "   ")
        
        print()'''

    row = []
    column = []
    TP = []
    FN = []
    FP = []
    TN = []
    for a in real:
        c = 0
        for b in real:
            c += real[a][b]
            if a == b:
                TP.append(real[a][b])
        row.append(c)
    accuracy = 0
    for a in real:
        c = 0
        for b in real:
            c += real[b][a]
        column.append(c)
    for a, b in zip(row, TP):
        accuracy += b/a
        FN.append(a-b)
    for a, b in zip(column, TP):
        FP.append(a-b)
    total = sum(row)
    for a,b,c in zip (TP, FP, FN):
        TN.append(total - a - b - c)
    accuracy /= 13
    print(accuracy)

    recall = []
    precision = []
    for tp, fp in zip(TP,FP):
        precision.append(round(tp/(tp+fp), 3))

    for tp, fn in zip(TP,FN):
        recall.append(round(tp/(tp+fn), 3))
    
    print(TP)
    print("Precision")
    print(precision)
    print("Recall")
    print(recall)
    print("Accuracy of BB")
    print(bb_accuracy/sum(TP))
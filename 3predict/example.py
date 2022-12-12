predicted = {
	"long sleeve dress"     :0,
    "long sleeve outwear"   :0,
    "long sleeve top"       :0,
    "short sleeve dress"    :0,
    "short sleeve outwear"  :0,
    "short sleeve top"      :0,
    "short"                :0,
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
    "short"                :1,
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
    "short"                :predicted.copy(),
    "skirt"                 :predicted.copy(),
    "sling"                 :predicted.copy(),
    "sling dress"           :predicted.copy(),
    "trousers"              : predicted.copy(),
    "vest"                  : predicted.copy(),
    "vest dress"            :predicted.copy(),
}

if __name__ == "__main__":
    filein = open("ciao.txt", "r")

    while(filein):
        line = filein.readline()
        if line == "end": break

        vals = line.split(" ")
        
        nums = vals[2]

        for i in range(int(nums)):
            readpred = filein.readline().split(",")

            realval = readpred[0].strip()
            
            items = [readpred[1].strip(), readpred[2].strip(), readpred[3].strip()]
            
            if realval in items:
                real[realval][realval] += 1
            else:
                position = category[realval]
                wrongprediction = items[position]

                real[realval][wrongprediction] += 1
                
        
    for key, val in real.items():
        print(f"{key:30}", end = " ")
        for kk, vv in val.items():
            print(f"{vv:5}", end = "   ")
        
        print()
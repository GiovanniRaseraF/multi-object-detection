import torch
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt

a = [0.9, 0.1, 0.9, 0.4]
c = [0.1, 0.4, 0.9, 0.4]
b = [0.4, 0-1, 0.6, 0.3]
a = torch.Tensor(a)
b = torch.Tensor(b)

#print(tv.ops.distance_box_iou_loss(a, b) + tv.ops.distance_box_iou_loss(c, b))



import torch
pred_label = [0.8, 0.04, 0.45, 0.58]
target_label = [1, 0, 0, 0]
pred_label = torch.Tensor(pred_label)
target_label = torch.Tensor(target_label)
print((-pred_label.log() * target_label).sum(dim=0).mean())
print(F.cross_entropy(pred_label, target_label))

listone = []
list = [1,2,3,4,5,6,7,8]
for i, z_ in enumerate(list):
    if i%4 == 0:
        list2 = list[i:i+4]
        listone.append(list2)

print(listone)
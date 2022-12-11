import torch
import torch.nn.functional as F
import torchvision as tv
a = [0.9, 0.1, 0.9, 0.4]
c = [0.1, 0.4, 0.9, 0.4]
b = [0.4, 0-1, 0.6, 0.3]
a = torch.Tensor(a)
b = torch.Tensor(b)

print(tv.ops.distance_box_iou_loss(a, b) + tv.ops.distance_box_iou_loss(c, b))
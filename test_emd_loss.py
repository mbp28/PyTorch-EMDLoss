import torch

import time

from emd import EMDLoss

dist =  EMDLoss()
torch.manual_seed(0)
p1 = torch.rand(3,5,3).cuda().float()
p2 = torch.rand(3,10,3).cuda().float()
p1.requires_grad = True
p2.requires_grad = True

s = time.time()
cost = dist(p1, p2)
emd_time = time.time() - s

print('Time: ', emd_time)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(p1.grad)
print(p2.grad)
print('Check', dist(p1, p1), dist(p2,p2))
# too big to run
a = torch.randn(1, 30000, 3).cuda()
b = torch.randn(1, 30000, 3).cuda()
cost = dist(a, b)
print('Success')

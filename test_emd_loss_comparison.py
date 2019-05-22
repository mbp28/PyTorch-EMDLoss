import torch

import time
from pdb import set_trace
from emd import EMDLoss
from scipy.stats import wasserstein_distance as wd
from pyemd import emd, emd_samples
import numpy as np
import cv2

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig

dist =  EMDLoss()
torch.manual_seed(0)
p1 = torch.rand(1,5,1).cuda().float()
p2 = torch.rand(1,5,1).cuda().float()
# p1 = torch.rand(1,5,1).cuda().float()
# p2 = torch.rand(1,5,1).cuda().float()
p1.requires_grad = True
p2.requires_grad = True
print('check')
s = time.time()
cost = dist(p1, p2)
emd_time = time.time() - s

print('Time: ', emd_time)
print(cost)

# wd
a1 = p1.cpu().detach().numpy().reshape(5)
a2 = p2.cpu().detach().numpy().reshape(5)
print(wd(a1,a2))
# pyemd
w1 = np.ones((5,5))
a1 = p1.cpu().detach().numpy().reshape(5).astype('float64')
a2 = p2.cpu().detach().numpy().reshape(5).astype('float64')
print(emd(a1,a2,w1))
# cv2
a1 = p1.cpu().detach().numpy().reshape(5,1)
a2 = p2.cpu().detach().numpy().reshape(5,1)
sig1 = img_to_sig(a1)
sig2 = img_to_sig(a2)
dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)
print(dist)
set_trace()
# loss = torch.sum(cost)
# print(loss)
# loss.backward()
# print(p1.grad)
# print(p2.grad)
# print('Check', dist(p1, p1), dist(p2,p2))
# # too big to run
# s = time.time()
# a = torch.randn(4, 30000, 3).cuda()
# b = torch.randn(4, 30000, 3).cuda()
# cost = dist(a, b)
# emd_time2 = time.time() - s
# print(emd_time2)
# print('Success')

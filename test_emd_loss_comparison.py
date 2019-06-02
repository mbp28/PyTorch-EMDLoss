import torch
import time
import cv2
import numpy as np
from pdb import set_trace
from emd import EMDLoss as cuda_emd
from scipy.stats import wasserstein_distance as scipy_emd #scipy
from pyemd import emd as py_emd #github python version
from cv2 import EMD as cv_emd #openCV

# Generate data with numpy
n1 = 5 # number of points in first set
n2 = 5 # number of points in second set
dim = 1
pts1 = np.random.randn(n1, dim)
pts2 = np.random.randn(n2, dim)

# Scipy EMD
if dim == 1:
    # scipy only works on univariate data
    scipy_loss = scipy_emd(pts1.squeeze(), pts2.squeeze())
    print("Scipy EMD {:.4f}".format(scipy_loss))

# PyEMD
# each point becomes a histogram bin, each point set becomes a binary vector to
# indicate which bins (i.e. points) it contains # use pairwise distances
# between histogram bins to get the correct emd
pts = np.concatenate([pts1, pts2])
dst = scipy.spatial.distance_matrix(pts, pts)
hist1 = (1 / n1) * np.concatenate([np.ones(n1), np.zeros(n2)])
hist2 = (1 / n2) * np.concatenate([np.zeros(n1), np.ones(n2)])
py_loss = py_emd(hist1, hist2, dst)
print("PyEMD {:.4f}".format(py_loss))

# OpenCV
# each signature is a matrix, first column gives weight (should be uniform for
# our purposes) and remaining columns give point coordinates, transformation
# from pts to sig is through function pts_to_sig
def pts_to_sig(pts):
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((pts.shape[0], 1 + pts.shape[1]), dtype=np.float32)
    sig[:,0] = (np.ones(pts.shape[0]) / pts.shape[0])
    sig[:,1:] = pts
    return sig
sig1 = pts_to_sig(pts1)
sig2 = pts_to_sig(pts2)
cv_loss, _, flow = cv_emd(sig1, sig2, cv2.DIST_L2)
print("OpenCV EMD {:.4f}".format(cv_loss))

# CUDA_EMD
pts1_cuda = torch.from_numpy(pts1).cuda().float().reshape(1, n1, dim)
pts2_cuda = torch.from_numpy(pts2).cuda().float().reshape(1, n2, dim)
pts1_cuda.requires_grad = True
pts2_cuda.requires_grad = True
cuda_loss = cuda_emd()(pts1_cuda, pts2_cuda)
print("CUDA EMD {:.4f}".format(cuda_loss))


# dist =  EMDLoss()
# torch.manual_seed(0)
# p1 = torch.rand(1,5,1).cuda().float()
# p2 = torch.rand(1,5,1).cuda().float()
# # p1 = torch.rand(1,5,1).cuda().float()
# # p2 = torch.rand(1,5,1).cuda().float()
# p1.requires_grad = True
# p2.requires_grad = True
# print('check')
# s = time.time()
# cost = dist(p1, p2)
# emd_time = time.time() - s
#
# print('Time: ', emd_time)
# print(cost)
#
# # wd
# a1 = p1.cpu().detach().numpy().reshape(5)
# a2 = p2.cpu().detach().numpy().reshape(5)
# print(wd(a1,a2))
# # pyemd
# w1 = np.ones((5,5))
# a1 = p1.cpu().detach().numpy().reshape(5).astype('float64')
# a2 = p2.cpu().detach().numpy().reshape(5).astype('float64')
# print(emd(a1,a2,w1))
# # cv2
# a1 = p1.cpu().detach().numpy().reshape(5,1)
# a2 = p2.cpu().detach().numpy().reshape(5,1)
# sig1 = img_to_sig(a1)
# sig2 = img_to_sig(a2)
# dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)
# print(dist)

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

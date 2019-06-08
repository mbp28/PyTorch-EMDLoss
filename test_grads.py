import torch
import argparse
import time
import cv2
import numpy as np
from pdb import set_trace
from emd import EMDLoss as cuda_emd
from cv2 import EMD as cv_emd #openCV

def main(n1, n2, dim, seed):
    # Generate data with numpy
    np.random.seed(seed)
    pts1 = np.random.randn(n1, dim)
    pts2 = np.random.randn(n2, dim)
    grad_ix_n = np.random.randint(min(n1, n2), size=5)
    grad_ix_dim = np.random.randint(dim, size=5)

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
    print("CUDA EMD (raw dim) {:.4f}".format(cuda_loss.item()))
    # and gradients
    cuda_loss.backward()
    pts1_grad_np = pts1_cuda.grad.cpu().numpy()
    pts2_grad_np = pts2_cuda.grad.cpu().numpy()
    print("CUDA EMD Grad t1 (mean) {:.4f}".format(pts1_grad_np.mean()))
    print("CUDA EMD Grad t1 (std) {:.4f}".format(pts1_grad_np.std()))
    print("CUDA EMD Grad t2 (mean) {:.4f}".format(pts2_grad_np.mean()))
    print("CUDA EMD Grad t2 (std) {:.4f}".format(pts2_grad_np.std()))
    print("CUDA EMD Grad t1 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
        *pts1_grad_np[0, grad_ix_n, grad_ix_dim]))
    print("CUDA EMD Grad t2 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
        *pts2_grad_np[0, grad_ix_n, grad_ix_dim]))

    if dim <= 6:
        pts1_cuda = torch.zeros(1, n1, 6).cuda().float()
        pts1_cuda[:, :, :dim] =  torch.from_numpy(pts1).cuda().float().reshape(1, n1, dim)
        pts2_cuda = torch.zeros(1, n2, 6).cuda().float()
        pts2_cuda[:, :, :dim] =  torch.from_numpy(pts2).cuda().float().reshape(1, n2, dim)
        pts1_cuda.requires_grad = True
        pts2_cuda.requires_grad = True
        cuda_loss = cuda_emd()(pts1_cuda, pts2_cuda)
        print("CUDA EMD (six adjusted) {:.4f}".format(cuda_loss.item()))
        # and gradients
        cuda_loss.backward()
        pts1_grad_np = pts1_cuda.grad.cpu().numpy()[:, :, :dim]
        pts2_grad_np = pts2_cuda.grad.cpu().numpy()[:, :, :dim]
        print("CUDA EMD Grad t1 (mean) {:.4f}".format(pts1_grad_np.mean()))
        print("CUDA EMD Grad t1 (std) {:.4f}".format(pts1_grad_np.std()))
        print("CUDA EMD Grad t2 (mean) {:.4f}".format(pts2_grad_np.mean()))
        print("CUDA EMD Grad t2 (std) {:.4f}".format(pts2_grad_np.std()))
        print("CUDA EMD Grad t1 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
            *pts1_grad_np[0, grad_ix_n, grad_ix_dim]))
        print("CUDA EMD Grad t2 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
            *pts2_grad_np[0, grad_ix_n, grad_ix_dim]))

    # set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n1', type=int, default=5)
    parser.add_argument('-n2', type=int, default=5)
    parser.add_argument('-dim', type=int, default=1)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    main(args.n1, args.n2, args.dim, args.seed)

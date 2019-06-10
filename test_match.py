import torch
import argparse
import time
import numpy as np
from pdb import set_trace
from emd import EMDLoss as cuda_emd
from _emd_ext._emd import emd_forward, emd_backward

def main(n, dim, seed):
    # Generate data with numpy
    np.random.seed(seed)
    pts1 = np.random.randn(n, dim)
    pts2 = np.random.randn(n, dim)
    # grad_ix_n = np.random.randint(n, size=5)
    # grad_ix_dim = np.random.randint(dim, size=5)

    # CUDA_EMD
    print('*** Computing cost and match for point clouds with {} points - no scaling'.format(n))
    pts1_cuda = torch.from_numpy(pts1).cuda().float().reshape(1, n, dim)
    pts2_cuda = torch.from_numpy(pts2).cuda().float().reshape(1, n, dim)
    cost, match = emd_forward(pts1_cuda, pts2_cuda)
    match_np = match.cpu().numpy()
    print('Cost {:.4f}, per point {:.4f}'.format(cost.item(), cost.item() / n))
    print('Statistics for match')
    numel = np.size(match_np)
    zero_numel = (match_np == 0).sum()
    close_zero_numel = np.isclose(match_np).sum()
    avg = (1 / n)
    above_avg_numel = (match_np > avg).sum()
    above_01_numel = (match_np > 0.1).sum()
    above_05_numel = (match_np > 0.5).sum()
    n_one_match = np.any(np.isclose(match_np, 1), 1).sum()
    n_sig_match = np.any(match_np > 0.5, 1).sum()
    n_some_match = np.any(match_np > 0.1, 1).sum()
    print('Zero elements, total {1}/{0}, average per point {2}'.format(
        numel, zero_numel, zero_numel / n))
    print('Close to zero elements, total {1}/{0}, average per point {2}'.format(
        numel, close_zero_numel, close_zero_numel / n))
    print('> avg elements, total {1}/{0}, average per point {2}'.format(
        numel, above_avg_numel, above_avg_numel / n))
    print('> 0.1 elements, total {1}/{0}, average per point {2}'.format(
        numel, above_01_numel, above_01_numel / n))
    print('> 0.5 elements, total {1}/{0}, average per point {2}'.format(
        numel, above_05_numel, above_05_numel / n))
    print('Number of points matched exactly, total {1}/{0} [{2:.3f}]'.format(
        n, n_one_match, n_one_match/n))
    print('Number of points matched significantly, total {1}/{0} [{2:.3f}]'.format(
        n, abovn_sig_matche_05, n_sig_match/n))
    print('Number of points matched somewhat, total {1}/{0} [{2:.3f}]'.format(
        n, n_some_match, n_some_match/n))

    # CUDA_EMD
    print('*** Computing cost and match for point clouds with {} points - scaling inputs with {}'.format(
        n, 1/n))
    pts1_cuda = torch.from_numpy(pts1).cuda().float().reshape(1, n, dim)
    pts2_cuda = torch.from_numpy(pts2).cuda().float().reshape(1, n, dim)
    cost, match = emd_forward(pts1_cuda, pts2_cuda)
    match_np = match.cpu().numpy()
    print('Cost {:.4f}, per point {:.4f}'.format(cost.item() * n, cost.item()))
    print('Statistics for match')
    numel = np.size(match_np)
    zero_numel = (match_np == 0).sum()
    close_zero_numel = np.isclose(match_np).sum()
    avg = (1 / n)
    above_avg_numel = (match_np > avg).sum()
    above_01_numel = (match_np > 0.1).sum()
    above_05_numel = (match_np > 0.5).sum()
    n_one_match = np.any(np.isclose(match_np, 1), 1).sum()
    n_sig_match = np.any(match_np > 0.5, 1).sum()
    n_some_match = np.any(match_np > 0.1, 1).sum()
    print('Zero elements, total {1}/{0}, average per point {2}'.format(
        numel, zero_numel, zero_numel / n))
    print('Close to zero elements, total {1}/{0}, average per point {2}'.format(
        numel, close_zero_numel, close_zero_numel / n))
    print('> avg elements, total {1}/{0}, average per point {2}'.format(
        numel, above_avg_numel, above_avg_numel / n))
    print('> 0.1 elements, total {1}/{0}, average per point {2}'.format(
        numel, above_01_numel, above_01_numel / n))
    print('> 0.5 elements, total {1}/{0}, average per point {2}'.format(
        numel, above_05_numel, above_05_numel / n))
    print('Number of points matched exactly, total {1}/{0} [{2:.3f}]'.format(
        n, n_one_match, n_one_match/n))
    print('Number of points matched significantly, total {1}/{0} [{2:.3f}]'.format(
        n, abovn_sig_matche_05, n_sig_match/n))
    print('Number of points matched somewhat, total {1}/{0} [{2:.3f}]'.format(
        n, n_some_match, n_some_match/n))


    # set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    main(args.n, args.dim, args.seed)

import torch
import time
import numpy as np
import cv2
from cv2 import EMD as cv_emd #openCV
from pdb import set_trace
from emd import EMDLoss as cuda_emd
from _emd_ext._emd import emd_forward, emd_backward

output = torch.load('data/out.pt').cuda()
target = torch.load('data/targ.pt').cuda()
pts1_cuda = output[0,:,:].reshape(1, output.size(1), output.size(2))
pts2_cuda = target[0,:,:].reshape(1, target.size(1), target.size(2))

for i in range(0, output.size(0)):
    print("min", output[i].min(0), target[i].min(0))
    print("mean", output[i].mean(0), target[i].mean(0))
    print("std", output[i].std(0), target[i].std(0))
    print("max", output[i].max(0), target[i].max(0))

    # CUDA_EMD
def standardise1(t):
    s = ((t - t.mean(1)) / t.std(1))
    return s

def standardise2(t):
    min, _ = t.min(1)
    range = t.max(1)[0] - min
    s = (t- min) / range
    return s

def standardise3(t):
    min  = t.min()
    range = t.max() - min
    s = (t - min) / range
    return s

def transform1(t):
    return torch.sign(t) * torch.log(1+t.abs())

def transform2(t):
    return torch.sign(t) * t.pow(2)

def run_match(t1, t2, scale):
    n = t1.size(1)
    print('*** Computing cost and match for point clouds with {} points - scaling inputs with {}'.format(n, scale))
    cost, match = emd_forward(t1 * scale, t2 * scale)
    match_np = match.cpu().numpy()
    print('Cost {:.4f}, per point {:.4f}'.format(cost.item() * (1/scale), cost.item() * (1/scale) / n))
    print('Statistics for match')
    numel = np.size(match_np)
    zero_numel = (match_np == 0).sum()
    close_zero_numel = np.isclose(match_np, 0).sum()
    avg = (1 / n)
    above_avg_numel = np.isclose(match_np, avg).sum() #(match_np > avg).sum()
    above_01_numel = (match_np > 0.1).sum()
    above_05_numel = (match_np > 0.5).sum()
    n_one_match = np.any(np.isclose(match_np, 1), 1).sum()
    n_sig_match = np.any(match_np > 0.5, 1).sum()
    n_some_match = np.any(match_np > 0.1, 1).sum()
    print('= 0.0 elements: \t Total {1:.3g}/{0:.3g} \t ({2:.3f} per point)'.format(
        numel, zero_numel, zero_numel / n))
    print('~ 0.0 elements: \t Total {1:.3g}/{0:.3g} \t ({2:.3f} per point)'.format(
        numel, close_zero_numel, close_zero_numel / n))
    print('~ avg elements: \t Total {1:.3g}/{0:.3g} \t ({2:.3f} per point)'.format(
        numel, above_avg_numel, above_avg_numel / n))
    print('> 0.1 elements: \t Total {1:.3g}/{0:.3g} \t ({2:.3f} per point)'.format(
        numel, above_01_numel, above_01_numel / n))
    print('> 0.5 elements: \t Total {1:.3g}/{0:.3g} \t ({2:.3f} per point)'.format(
        numel, above_05_numel, above_05_numel / n))
    print('Exact matches:    \t Total {1:.3g}/{0:.3g} \t [{2:.2%}]'.format(
        n, n_one_match, n_one_match/n))
    print('Significant matches: \t Total {1:.3g}/{0:.3g} \t [{2:.2%}]'.format(
        n, n_sig_match, n_sig_match/n))
    print('Somewhat matches: \t Total {1:.3g}/{0:.3g} \t [{2:.2%}]'.format(
        n, n_some_match, n_some_match/n))

a = torch.from_numpy(np.random.randn(1, 10000, 3)).cuda()
b = a.float() + torch.FloatTensor([10.0,0, 0]).cuda()
print('numpy dispersion mean')
run_match(a.float(), b, 1)

#for i in range(output.shape[0]):
#    run_match(standardise2(output[i].reshape(1, output.shape[1], output.shape[2])), standardise2(target[i].reshape(1, output.shape[1], output.shape[2])), 1)

#for i in range(output.shape[0]):
#    run_match(standardise3(output[i].reshape(1, output.shape[1], output.shape[2])), standardise3(target[i].reshape(1, output.shape[1], output.shape[2])), 1)

#for i in range(output.shape[0]):
#    run_match(transform2(output[i].reshape(1, output.shape[1], output.shape[2])), transform2(target[i].reshape(1, output.shape[1], output.shape[2])), 1)

#print('Transform 1 only output')
#for i in range(output.shape[0]):
#    run_match(transform1(output[i].reshape(1, output.shape[1], output.shape[2])), target[i].reshape(1, output.shape[1], output.shape[2]), 1)

#print('Transform 2 only output')
#for i in range(output.shape[0]):
#    run_match(transform2(output[i].reshape(1, output.shape[1], output.shape[2])), target[i].reshape(1, output.shape[1], output.shape[2]), 1)

#n = 2000
#torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)
#for i in range(output.shape[0]):
#    perm = torch.randperm(output.size(1))
#    ix = perm[:n]
#    run_match(output[i,ix,:].reshape(1, n, output.shape[2]), target[i,ix,:].reshape(1, n, output.shape[2]), 1)


#for i in range(-10, 11):
#   scale = 10**i
#   run_match(pts1_cuda, pts2_cuda, scale)
#print('Output loop')
#for i in range(output.shape[0]):
#    run_match(output[i].reshape(1, output.shape[1], output.shape[2]), output[i].reshape(1, output.shape[1], output.shape[2]), 1)

#print('Target loop')
#for i in range(output.shape[0]):
#    run_match(target[i].reshape(1, output.shape[1], output.shape[2]), target[i].reshape(1, output.shape[1], output.shape[2]), 1)

#print('Pairwise loop')
#for i in range(output.shape[0]):
#    run_match(output[i].reshape(1, output.shape[1], output.shape[2]), target[i].reshape(1, output.shape[1], output.shape[2]), 1)

set_trace()

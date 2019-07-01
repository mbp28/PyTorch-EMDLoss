import torch
import torch.nn as nn
import time
import numpy as np
from pdb import set_trace
from emd import EMDLoss as cuda_emd
from _emd_ext._emd import emd_forward, emd_backward

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

class PointLayer(nn.Module):

    def __init__(self, points):
        super(PointLayer, self).__init__()
        self.points = nn.Parameter(points)

    def forward(self):
        return self.points

def ex1_random_clouds():
    n = 100
    x = torch.randn(n, 3).view(1, n, 3)
    L = torch.Tensor([[1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 0.5]]) # some var decomposition
    y = (torch.randn(n, 3).mm(L.t()) + torch.Tensor([1,2, 3])).view(1,n,3) # some mean
    run_match(x.cuda(), y.cuda(), 1)
    # set up training
    model = PointLayer(x)
    model = model.cuda()
    y = y.cuda()
    num_iter = 1000
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    # training loop
    for i in range(num_iter):
        optimiser.zero_grad()
        x = model() # just return the current points
        #x.register_hook(lambda grad: (1/n) * grad)
        loss = cuda_emd()(x, y)
        loss.backward()
        optimiser.step()
        if (i % 10) == 0:
            print("Iteration {} [{:.0%}]	Loss {:.4f} [{:.4f}]".format(i, i/num_iter, loss.item(), loss.item() / n))
        if ((i+1) % 99) == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] *= 0.5

    run_match(model.points, y.cuda(), 1)
    torch.save(model.points.data.cpu(), 'trained_point.pth')



def ex2_data():
    output = torch.load('data/out.pt').cuda()
    target = torch.load('data/targ.pt').cuda()
    n = target.size(1)
    x = output[0,:,:].reshape(1, output.size(1), output.size(2))
    y = target[0,:,:].reshape(1, target.size(1), target.size(2))
    run_match(x, y, 1)
    # set up training
    model = PointLayer(x)
    model = model.cuda()
    num_iter = 1000
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)
    # training loop
    for i in range(num_iter):
        optimiser.zero_grad()
        x = model() # just return the current points
        #x.register_hook(lambda grad: (1/n) * grad)
        loss = cuda_emd()(x, y)
        loss.backward()
        optimiser.step()
        if (i % 10) == 0:
            print("Iteration {} [{:.0%}]	Loss {:.4f} [{:.4f}]".format(i, i/num_iter, loss.item(), loss.item() / n))
        if ((i+1) % 99) == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] *= 0.9
                torch.save(model.points.data.cpu(), 'trained_point.pt')

    run_match(model.points, y.cuda(), 1)
    torch.save(model.points.data.cpu(), 'data/trained_points.pth')

if __name__ == '__main__':
    # ex1_random_clouds()
    ex2_data()

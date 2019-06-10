import torch

class CustomGradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(scale)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * scale, None

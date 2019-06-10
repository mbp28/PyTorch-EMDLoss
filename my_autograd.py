import torch

class CustomGradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        # scale = torch.empty_like(input).fill_(scale) # not sure if this is the best way
        # ctx.save_for_backward(scale)
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # scale, = ctx.saved_tensors
        scale = ctx.scale
        return grad_output * scale, None

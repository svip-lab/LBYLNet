import torch
import numpy as np
import landmarks
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from landmarkconv import _C

class TopLeftPoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.tl_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.tl_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class TopRightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.tr_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.tr_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class BottomRightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.br_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide = _C.br_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class BottomLeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.bl_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.bl_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class TopLeftPool(nn.Module):
    def forward(self, x, guide):
        x = x.contiguous()
        guide = guide.expand_as(x).contiguous()
        return TopLeftPoolFunction.apply(x, guide)

class TopRightPool(nn.Module):
    def forward(self, x, guide):
        x = x.contiguous()
        guide = guide.expand_as(x).contiguous()
        return TopRightPoolFunction.apply(x, guide)

class BottomRightPool(nn.Module):
    def forward(self, x, guide):
        x = x.contiguous()
        guide = guide.expand_as(x).contiguous()
        return BottomRightPoolFunction.apply(x, guide)

class BottomLeftPool(nn.Module):
    def forward(self, x, guide):
        x = x.contiguous()
        guide = guide.expand_as(x).contiguous()
        return BottomLeftPoolFunction.apply(x, guide)
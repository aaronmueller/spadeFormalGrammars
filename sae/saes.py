import torch
from sparsemax import Sparsemax
import torch.nn as nn
import torch.nn.functional as F
import torch

class SAE(torch.nn.Module):
    def __init__(self, dimin=2, width=5, sae_type='relu', kval_topk=None, 
                 normalize_decoder=False, lambda_init = None):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'jumprelu', 'sparsemax_lintx', 'sparsemax_dist'
        kval_topk: (int)
            k in topk sae_type
        normalize_decoder: (bool)
            whether to normalize the decoder weights to unit norm
        """
        super(SAE, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.normalize_decoder = normalize_decoder

        ## Encoder parameters
        self.be = nn.Parameter(torch.zeros((1, width)))
        self.Ae = nn.Parameter(torch.randn((width, dimin))) #N(0,1) init

        ## Decoder parameters
        self.bd = nn.Parameter(torch.zeros((1, dimin)))
        self.Ad = nn.Parameter(torch.randn((dimin, width))) #N(0,1) init
        with torch.no_grad():
            self.Ad.copy_(self.Ae.T) #at init, decoder is the transpose of encoder

        ## Parameters for specific SAEs
        # JumpReLU
        if sae_type=='jumprelu':
            self.logthreshold = nn.Parameter(torch.log(1e-3*torch.ones((1, width))))
            self.bandwidth = 1e-3 #width of rectangle used in approx grad of jumprelu wrt threshold

        # Sparsemax
        if 'sparsemax' in sae_type:
            lambda_init = 1/(width*dimin) if lambda_init is None else lambda_init
            lambda_pre = softplus_inverse(lambda_init)
            self.lambda_pre = nn.Parameter(lambda_pre) #trainable parameter (~inv temp) for sparsemax
        else:
            lambda_init = 1/(4*dimin) if lambda_init is None else lambda_init
            lambda_pre = softplus_inverse(lambda_init)
            self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False) #not trainable
        
        # Topk parameter
        if sae_type=='topk':
            if kval_topk is not None:
                self.kval_topk = kval_topk
            else:
                raise ValueError('kval_topk must be provided for topk sae_type')


    @property
    def lambda_val(self): #lambda_val is lambda, forced to be positive here
        return F.softplus(self.lambda_pre)


    def forward(self, x, return_hidden=False):
        lam = self.lambda_val

        if self.sae_type=='relu':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T) + self.be
            xint = F.relu(lam*x)
            if self.normalize_decoder:
                eps = 1e-6
                Ad_unit = self.Ad / (eps+torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        elif self.sae_type=='topk':
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T)
            a = torch.topk(F.relu(x), self.kval_topk, dim=-1)
            # _, topk_indices = torch.topk(F.relu(x), self.kval_topk, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, a[1], 1)
            xint = x * mask* lam
            if self.normalize_decoder:
                eps = 1e-6
                Ad_unit = self.Ad / (eps + torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        elif self.sae_type=='jumprelu':
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(lam*x)
            threshold = torch.exp(self.logthreshold)
            xint = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(xint, self.Ad.T) + self.bd
            if self.normalize_decoder:
                eps = 1e-6
                Ad_unit = self.Ad / (eps+torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd
                
        elif self.sae_type=='sparsemax_lintx':
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            sm = Sparsemax(dim=-1)
            xint = sm(lam*x)
            x = torch.matmul(xint, self.Ad.T) + self.bd

        elif self.sae_type=='sparsemax_dist':
            x = -lam*torch.square(torch.norm(x.unsqueeze(1)-self.Ae.unsqueeze(0), dim=-1))
            sm = Sparsemax(dim=-1)
            xint = sm(x)
            x = torch.matmul(xint, self.Ad.T)

        else:
            raise ValueError('Invalid sae_type')

        if not return_hidden:
            return x
        else:
            return x, xint



######################## UTILS ########################
def rectangle(x):
    # rectangle function
    return ((x >= -0.5) & (x <= 0.5)).float()

class JumpReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device="cuda")
        ctx.save_for_backward(x, threshold, bandwidth)
        return x*(x>threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors

        # Compute gradients
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)  # Aggregating across batch dimension
        
        return x_grad, threshold_grad, None  # None for bandwidth since const

def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)



def softplus_inverse(input, beta=1.0, threshold=20.0):
        """"
        inverse of the softplus function in torch
        """
        if isinstance(input, float):
                input = torch.tensor([input])
        if input*beta<threshold:
                return (1/beta)*torch.log(torch.exp(beta*input)-1.0)
        else:
              return input[0]


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=input.dtype, device=input.device)
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0*grad_output #no ste to input
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None  # None for bandwidth since const

def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)
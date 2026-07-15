import torch
from torch.autograd import Function

"""
This script contains the customized forward and backward of each element-wise activation function. 
"""

class cosh_parametric(Function):
    @staticmethod
    def forward(ctx, X, a):
        Z = a * X
        output = torch.cosh(Z)
        ctx.save_for_backward(X, Z, a)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, Z, a = ctx.saved_tensors
        
        # Grad wrt X 
        grad_X = grad_output * torch.sinh(Z) * a

        # Grad wrt a
        grad_a = torch.sum(grad_output * torch.sinh(Z) * X)

        return grad_X, grad_a



class cosh_parametric_tracenorm(Function):
    @staticmethod
    def forward(ctx, X, a):
        Z = a * X
        C = torch.cosh(Z)

        t = torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)
        c = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        s = t / c

        output = s.unsqueeze(-1).unsqueeze(-1) * C

        ctx.save_for_backward(X, Z, a)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, Z, a = ctx.saved_tensors

        C = torch.cosh(Z)

        t = torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)
        c = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        s = t / c

        # Frobenius inner product <grad_output, C>_F, per batch element
        inner_prod = torch.sum(grad_output * C, dim=(-2, -1))

        # Grad wrt X
        # Local term : s * alpha * sinh(Z), elementwise, scaled by upstream grad
        s_bc = s.unsqueeze(-1).unsqueeze(-1)
        local_term = s_bc * a * torch.sinh(Z) * grad_output

        # Global term : diagonal matrix D with D_ii = 1/c - t * alpha * sinh(Z_ii) / c^2,
        # scaled by the Frobenius inner product <grad_output, C>_F
        sinhZ_diag = torch.diagonal(torch.sinh(Z), dim1=-2, dim2=-1)
        c_bc = c.unsqueeze(-1)
        t_bc = t.unsqueeze(-1)
        D_diag = 1.0 / c_bc - t_bc * a * sinhZ_diag / (c_bc ** 2)
        D = torch.diag_embed(D_diag)

        global_term = D * inner_prod.unsqueeze(-1).unsqueeze(-1)

        grad_X = local_term + global_term

        # Grad wrt a
        # Local term : s * sum(grad_output * X * sinh(Z))
        grad_a_local = torch.sum(s * torch.sum(grad_output * X * torch.sinh(Z), dim=(-2, -1)))

        # Global term : -(t / c^2) * E * <grad_output, C>_F, with E = sum_i X_ii * sinh(Z_ii)
        X_diag = torch.diagonal(X, dim1=-2, dim2=-1)
        E = (X_diag * sinhZ_diag).sum(-1)
        grad_a_global = torch.sum(-(t / c ** 2) * E * inner_prod)

        grad_a = grad_a_local + grad_a_global

        return grad_X, grad_a



class sinh_parametric(Function):
    @staticmethod
    def forward(ctx, X, a):
        Z = a * X
        output = torch.sinh(Z)
        ctx.save_for_backward(X, Z, a)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, Z, a = ctx.saved_tensors
        
        # Grad wrt X 
        grad_X = grad_output * torch.cosh(Z)*a

        # Grad wrt a 
        grad_a = torch.sum(grad_output * torch.cosh(Z)*X)

        return grad_X, grad_a



class polynomialFunction(Function):
    @staticmethod
    def forward(ctx, X, coeffs):

        Y = coeffs[0] * torch.ones_like(X)

        H = torch.ones_like(X)

        for k in range(1, len(coeffs)):
            H = H * X
            Y = Y + coeffs[k] * H

        ctx.save_for_backward(X, coeffs)

        return Y

    @staticmethod
    def backward(ctx, grad_output):

        X, coeffs = ctx.saved_tensors
        K = len(coeffs) - 1

        # grad wrt x 
        grad_X = torch.zeros_like(X)

        H = torch.ones_like(X)  # X^(k-1)

        for k in range(1, K + 1):
            grad_X += k * coeffs[k] * H
            H = H * X

        grad_X = grad_X * grad_output

        # grad wrt coeffs 
        grad_coeffs = torch.zeros_like(coeffs)

        H = torch.ones_like(X)

        for k in range(K + 1):
            grad_coeffs[k] = torch.sum(grad_output * H)
            H = H * X

        return grad_X, grad_coeffs



class exp_parametric(Function):
    @staticmethod
    def forward(ctx, X, a):
        Z = a * X
        output = torch.exp(Z)
        ctx.save_for_backward(X, Z, a)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, Z, a = ctx.saved_tensors
        
        # Grad wrt X 
        grad_X = grad_output * torch.exp(Z)*a

        # Grad wrt a 
        grad_a = torch.sum(grad_output * torch.exp(Z)*X)

        return grad_X, grad_a



class truncated_exponential(Function):

    @staticmethod
    def forward(ctx, X, alpha, K):

        # construire les coefficients alpha^k / k!
        coeffs = []
        ak = 1.0

        for k in range(K + 1):
            if k > 0:
                ak = ak * alpha / k   
            coeffs.append(ak)

        # appliquer sur X 
        Y = coeffs[0] * torch.ones_like(X)

        H = torch.ones_like(X)

        for k in range(1, K + 1):
            H = H * X
            Y = Y + coeffs[k] * H

        ctx.save_for_backward(X, alpha)
        ctx.K = K
        ctx.coeffs = coeffs

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, alpha = ctx.saved_tensors
        K = ctx.K 
        coeffs = ctx.coeffs

        # grad wrt X 
        grad_X = torch.zeros_like(X)
        H = torch.ones_like(X)

        for k in range(1, K+1):
            grad_X = grad_X + grad_output * coeffs[k] * k * H
            H = H * X

        # grad alpha
        grad_alpha = 0.0
        H = torch.ones_like(X)

        for k in range(1, K + 1):
            H = H * X
            grad_alpha = grad_alpha + torch.sum(
                grad_output * H * (k * coeffs[k] / alpha)
            )

        return grad_X, grad_alpha, None
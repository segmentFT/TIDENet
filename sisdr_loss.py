import torch
import torch.nn as nn
import numpy as np



class si_sidrloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        y_true = target
        y_pred = pred
        x = y_true.squeeze()
        y = y_pred.squeeze()

        smallVal = 1e-9  # To avoid divide by zero
        a = torch.sum(y * x, dim=-1, keepdims=True) / (torch.sum(x * x, dim=-1, keepdims=True) + smallVal)

        xa = a * x
        xay = xa - y
        d = torch.sum(xa * xa, dim=-1, keepdims=True) / (torch.sum(xay * xay, dim=-1, keepdims=True) + smallVal)
        # d1=tf.zeros(d.shape)
        d1 = d == 0
        d1 = 1 - torch.tensor(d1, dtype=torch.float32)

        d = -torch.mean(10 * d1 * torch.log10(d + smallVal))
        return d

    # def forward(self, pred, target):

    #     x = pred.squeeze()
    #     y = target.squeeze()
        

    #     eps = 1e-8

    #     def l2norm(mat, keepdim=False):
    #         return torch.norm(mat, dim=-1, keepdim=keepdim)
        
    #     if x.shape != y.shape:
    #         raise RuntimeError(
    #             "Dimention mismatch when calculate si-snr, {} vs {}".format(x.shape, y.shape))
    #     x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    #     y_zm = y - torch.mean(y, dim=-1, keepdim=True)
    #     t = torch.sum(x_zm * y_zm, dim=-1, keepdim=True) * y_zm / (l2norm(y_zm, keepdim=True)**2 + eps)
    #     d = -torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
        
    #     return d
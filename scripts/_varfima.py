'''
Tools for simulation of VARMA and VARFIMA processes
Compatible with PyTorch 2.x
'''
import torch
import numpy as np


# VARMA sim
def sim_VARMA(T, k=1, VAR=None, VMA=None, cov=None, innov=None):
    if cov is None:
        cov = torch.eye(k)
        chol = torch.eye(k)
    else:
        chol = torch.linalg.cholesky(cov)

    if innov is None:
        innov = torch.matmul(chol, torch.randn((k, T)))

    if VAR is not None:
        p = VAR.shape[2]
    else:
        p = 0

    if VMA is not None:
        q = VMA.shape[2]
    else:
        q = 0

    u0 = innov
    if q > 0:  # moving average
        for i in range(q, T):
            for qq in range(1, q + 1):
                u0[:, i] += torch.matmul(VMA[:, :, qq - 1], u0[:, i - qq])

    if p > 0:  # vector autoregression
        for i in range(p, T):
            for pp in range(1, p + 1):
                u0[:, i] -= torch.matmul(VAR[:, :, pp - 1], u0[:, i - pp])

    return u0


# fractional differencing
def fracdiff(seq, d):
    """
    Takes k x T sequence, len-k d and returns row-wise fractionally differenced seq of same dim
    Updated for PyTorch 2.x FFT API
    """
    k, T = seq.shape
    seq_ext = torch.cat((seq, torch.zeros(k, T - 1)), dim=1)
    
    # Convertir en complexe (nouvelle API PyTorch)
    seq_ext_complex = torch.complex(seq_ext, torch.zeros_like(seq_ext))
    seq_ext_fft = torch.fft.fft(seq_ext_complex, dim=1)

    filt = torch.zeros((k, 2 * T - 1))
    for i in range(k):
        filt[i, :T] = fd_filter(d[i], T)
    
    # Convertir le filtre en complexe
    filt_complex = torch.complex(filt, torch.zeros_like(filt))
    filt_fft = torch.fft.fft(filt_complex, dim=1)

    # Multiplication de nombres complexes (PyTorch gère ça nativement)
    prod = filt_fft * seq_ext_fft

    # Transformée inverse et extraction de la partie réelle
    result = torch.fft.ifft(prod, dim=1)
    return result[:, :T].real


def fd_filter(d, T):
    filt = torch.cumprod((torch.arange(1, T, dtype=torch.float32) + d - 1) / torch.arange(1, T, dtype=torch.float32), 0)
    return torch.cat((torch.ones(1), filt))


# VARFIMA
def sim_VARFIMA(T, k, d, VAR, VMA, cov=None):
    skip = 2000

    if cov is None:
        cov = torch.eye(k)
        innov = torch.randn((k, T + 2 * skip))
    else:
        chol = torch.linalg.cholesky(cov)
        innov = torch.matmul(chol, torch.randn((k, T + 2 * skip)))

    if VAR is not None:
        p = VAR.shape[2]
    else:
        p = 0

    if VMA is not None:
        q = VMA.shape[2]
    else:
        q = 0

    seq = sim_VARMA(T + 2 * skip, k, VAR, VMA, cov, innov)
    seq = fracdiff(seq, d)

    CVMA = cov
    if q > 0:
        sum_MA = torch.sum(VMA, dim=2)
        CVMA = torch.matmul((torch.eye(k) + sum_MA), torch.matmul(CVMA, (torch.eye(k) + sum_MA).t()))

    CVAR = torch.eye(k)
    if p > 0:
        sum_AR = torch.sum(VAR, dim=2)
        CVAR = torch.inverse(torch.eye(k) + sum_AR)

    long_run_cov = torch.matmul(CVAR, torch.matmul(CVMA, CVAR.t()))

    return seq[:, -T:], long_run_cov


def sim_FD(T, k, d):
    """Wrapper for fractionally differenced Gaussian noise"""
    return sim_VARFIMA(T, k, d, None, None)
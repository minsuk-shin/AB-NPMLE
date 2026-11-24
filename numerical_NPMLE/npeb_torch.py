"""PyTorch versions of the NPEB kernel and sparse EM matrix operations.

The formulas and CSR/COO flow follow npeb.GLMixture. Both CPU and CUDA use
this module so their timing comparison changes only the tensor device.
"""

import numpy as np
import torch
import time


class SparseMatrix:
    """CSR matrix with the NPEB operations used by GLMixture.fit."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.layout != torch.sparse_csr:
            raise TypeError("expected a CSR tensor")
        self.tensor = tensor

    @property
    def shape(self):
        return self.tensor.shape

    def row_indices(self):
        counts = self.tensor.crow_indices()[1:] - self.tensor.crow_indices()[:-1]
        return torch.repeat_interleave(
            torch.arange(self.shape[0], device=counts.device), counts,
            output_size=self.tensor.values().numel())

    def multiply(self, weights: torch.Tensor):
        """Match scipy CSR.multiply(weights), which produces COO values."""
        columns = self.tensor.col_indices()
        return SparseProduct(self.row_indices(), columns,
                             self.tensor.values() * weights[columns], self.shape)

    def dot(self, other: torch.Tensor):
        if other.ndim == 1:
            return torch.sparse.mm(self.tensor, other[:, None]).reshape(-1)
        return torch.sparse.mm(self.tensor, other)

    def sum(self, axis: int):
        if axis != 0:
            raise ValueError("only the NPEB axis=0 sum is supported")
        result = torch.zeros(self.shape[1], dtype=self.tensor.dtype,
                             device=self.tensor.device)
        result.index_add_(0, self.tensor.col_indices(), self.tensor.values())
        return result[None, :]

    def transpose(self):
        return TransposedSparseMatrix(self)


class SparseProduct:
    """COO values returned by B.multiply(weights), before L1 normalization."""

    def __init__(self, rows, columns, values, shape):
        self.rows = rows
        self.columns = columns
        self.values = values
        self.shape = shape


class TransposedSparseMatrix:
    def __init__(self, matrix: SparseMatrix):
        self.matrix = matrix

    def dot(self, X: torch.Tensor):
        matrix = self.matrix
        result = torch.zeros((matrix.shape[1], X.shape[1]), dtype=X.dtype,
                             device=X.device)
        result.index_add_(0, matrix.tensor.col_indices(),
                          matrix.tensor.values()[:, None] * X[matrix.row_indices()])
        return result


def normalize(resp: SparseProduct, norm: str = 'l1', axis: int = 1,
              copy: bool = False) -> SparseMatrix:
    """Match sklearn.normalize(resp, norm='l1', axis=1) for nonnegative PDF data."""
    if norm != 'l1' or axis != 1:
        raise ValueError("only NPEB row-wise L1 normalization is supported")
    row_norm = torch.zeros(resp.shape[0], dtype=resp.values.dtype,
                           device=resp.values.device)
    row_norm.index_add_(0, resp.rows, resp.values.abs())
    divisors = row_norm[resp.rows]
    values = torch.where(divisors != 0, resp.values / divisors, resp.values)
    indices = torch.stack((resp.rows, resp.columns))
    coo = torch.sparse_coo_tensor(indices, values, resp.shape).coalesce()
    return SparseMatrix(coo.to_sparse_csr())


def log_mvn_pdf(X: torch.Tensor, atoms: torch.Tensor, prec: torch.Tensor,
                prec_type: str, homoscedastic: bool,
                profile_components=None) -> torch.Tensor:
    """Torch port of npeb.log_mvn_pdf with the same four precision cases."""
    d = X.shape[1]
    if prec_type == 'general' and not homoscedastic:
        diffs = X[:, None, :] - atoms[None, :, :]
        M = torch.einsum('ijk,ikl,ijl->ij', diffs, prec, diffs)
        ld = torch.linalg.slogdet(prec).logabsdet[:, None]
    elif prec_type == 'diagonal' and not homoscedastic:
        diffs = X[:, None, :] - atoms[None, :, :]
        M = torch.einsum('ik,ijk->ij', prec, diffs**2)
        ld = torch.sum(torch.log(prec), dim=1)[:, None]
    elif prec_type == 'general' and homoscedastic:
        M = (torch.einsum('ik,kl,il->i', X, prec, X)[:, None]
             + torch.einsum('jk,kl,jl->j', atoms, prec, atoms)[None, :]
             - 2 * torch.einsum('ik,kl,jl->ij', X, prec, atoms))
        ld = torch.linalg.slogdet(prec).logabsdet
    elif prec_type == 'diagonal' and homoscedastic:
        if profile_components is None:
            M = ((X**2 @ prec)[:, None]
                 + (atoms**2 @ prec)[None, :]
                 - 2 * torch.einsum('ik,k,jk->ij', X, prec, atoms))
            ld = torch.sum(torch.log(prec))
        else:
            t0 = time.perf_counter()
            term_x = (X**2 @ prec)[:, None]
            profile_components('X_squared_matmul_prec', t0)
            t0 = time.perf_counter()
            term_atoms = (atoms**2 @ prec)[None, :]
            profile_components('atoms_squared_matmul_prec', t0)
            t0 = time.perf_counter()
            cross = torch.einsum('ik,k,jk->ij', X, prec, atoms)
            profile_components('einsum_cross', t0)
            t0 = time.perf_counter()
            M = term_x + term_atoms - 2 * cross
            profile_components('combine_mahalanobis', t0)
            t0 = time.perf_counter()
            ld = torch.sum(torch.log(prec))
            profile_components('log_prec_sum', t0)
    else:
        raise ValueError("prec_type must be 'general' or 'diagonal'")
    if profile_components is None:
        return -(M + d * np.log(2 * np.pi) - ld) / 2
    t0 = time.perf_counter()
    result = -(M + d * np.log(2 * np.pi) - ld) / 2
    profile_components('form_log_pdf', t0)
    return result


def mvn_pdf(X: torch.Tensor, atoms: torch.Tensor, prec: torch.Tensor,
            prec_type: str, homoscedastic: bool, n_chunks: int = 1,
            log_prob_thresh: float = -float('Inf'),
            row_condition: bool = False, profile_components=None) -> SparseMatrix:
    """Torch port of npeb.mvn_pdf, including sparse conversion by chunk."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be positive")
    t0 = time.perf_counter() if profile_components is not None else None
    ind_list = torch.tensor_split(torch.arange(X.shape[0], device=X.device), n_chunks)
    if profile_components is not None:
        profile_components('split_indices', t0)
    res = []
    for inds in ind_list:
        t0 = time.perf_counter() if profile_components is not None else None
        chunk = X[inds]
        if profile_components is not None:
            profile_components('slice_X', t0)
        A = log_mvn_pdf(chunk, atoms,
                        prec if homoscedastic else prec[inds],
                        prec_type, homoscedastic, profile_components)
        if row_condition:
            A = A - A.max(dim=1, keepdim=True).values
        t0 = time.perf_counter() if profile_components is not None else None
        A = A.masked_fill(A < log_prob_thresh, -float('Inf'))
        if profile_components is not None:
            profile_components('threshold_mask', t0)
        t0 = time.perf_counter() if profile_components is not None else None
        pdf = A.exp()
        if profile_components is not None:
            profile_components('exp', t0)
        t0 = time.perf_counter() if profile_components is not None else None
        res.append(pdf.to_sparse_csr())
        if profile_components is not None:
            profile_components('dense_to_csr', t0)
    if len(res) == 1:
        t0 = time.perf_counter() if profile_components is not None else None
        result = SparseMatrix(res[0])
        if profile_components is not None:
            profile_components('stack_chunks', t0)
        return result
    t0 = time.perf_counter() if profile_components is not None else None
    offsets = []
    total = 0
    for part in res:
        offsets.append(part.crow_indices()[1:] + total)
        total += part.values().numel()
    crow = torch.cat([res[0].crow_indices()[:1], *offsets])
    columns = torch.cat([part.col_indices() for part in res])
    values = torch.cat([part.values() for part in res])
    result = SparseMatrix(torch.sparse_csr_tensor(
        crow, columns, values, size=(X.shape[0], atoms.shape[0]),
        device=X.device))
    if profile_components is not None:
        profile_components('stack_chunks', t0)
    return result


def score(X: torch.Tensor, atoms: torch.Tensor, weights: torch.Tensor,
          prec: torch.Tensor, prec_type: str = 'diagonal',
          homoscedastic: bool = True, n_chunks: int = 1,
          log_prob_thresh: float = -float('Inf'),
          profile_components=None) -> torch.Tensor:
    """Torch port of GLMixture.score, using sparse K.dot(weights)."""
    K = mvn_pdf(X, atoms, prec, prec_type, homoscedastic,
                n_chunks, log_prob_thresh, row_condition=False,
                profile_components=profile_components)
    if profile_components is None:
        return torch.log(K.dot(weights)).mean()
    t0 = time.perf_counter()
    weighted_pdf = K.dot(weights)
    profile_components('sparse_dot_weights', t0)
    t0 = time.perf_counter()
    logged = torch.log(weighted_pdf)
    profile_components('log', t0)
    t0 = time.perf_counter()
    result = logged.mean()
    profile_components('mean', t0)
    return result

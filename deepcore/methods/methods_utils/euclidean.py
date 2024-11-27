import torch
import numpy as np


def euclidean_dist(x, y):
    x = x.float()
    y = y.float()
    m, n = x.size(0), y.size(0)
    # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
    return dist


def cosine_similarity(A, B):
    # 计算 A 和 B 的 L2 范数
    norm_A = A.norm(dim=1, keepdim=True)  # (n1, 1)
    norm_B = B.norm(dim=1, keepdim=True)  # (n2, 1)

    # 计算 A 和 B 的点积
    similarity_matrix = torch.mm(A, B.t())  # (n1, m) @ (m, n2) -> (n1, n2)

    # 归一化
    cosine_sim = similarity_matrix / (norm_A * norm_B.t())  # (n1, n2)

    return -cosine_sim


def euclidean_dist_for_batch(x, y, batch=2048, metric=euclidean_dist):
    metric = cosine_similarity
    # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
    m, n = x.size(0), y.size(0)
    distance = torch.empty(m, n, dtype=x.dtype).to(device='cpu')
    i = 0
    while True:
        row_start = i * batch
        row_end = min((i + 1) * batch, m)
        distance[row_start:row_end] = metric(x[row_start:row_end], y).cpu()
        if row_end == m:
            break
        i = i + 1
    return (distance+1)/2


def euclidean_dist_pair(x):
    m = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)
    dist = xx + xx.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def euclidean_dist_np(x, y):
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    return np.sqrt(np.clip(x2 + y2 - 2. * xy, 1e-12, None))

def euclidean_dist_pair_np(x):
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

def fully_supervised_contrastive_loss(features, labels, group_labels, temperature):
    """
    Compute the supervised contrastive loss
    
    Args:
    - features: tensor of shape (batch_size, feature_dim)
    - labels: tensor of shape (batch_size,)
    - temperature: scalar value, temperature parameter
    
    Returns:
    - loss: scalar value, the supervised contrastive loss
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize feature vectors
    features = F.normalize(features, p=2, dim=1)
    
    # Compute pairwise cosine similarities
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create a mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    inter_mask = torch.ne(group_labels, group_labels.T).float().to(device)
    
    # Exclude self-contrast
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_exc = mask*logits_mask

    pos_inter_mask = mask*inter_mask
    neg_inter_mask = (1-mask)*inter_mask

    # Compute log_prob
    exp_logits = torch.exp(similarity_matrix)*logits_mask  
    neg_exp_logits = exp_logits*(1-mask) # all negatives

    log_prob = similarity_matrix - torch.log(exp_logits +  neg_exp_logits.sum(1, keepdim=True))

    neg_inter_exp_logits = exp_logits*neg_inter_mask  # inter negatives 
    log_prob_inter = similarity_matrix - torch.log(exp_logits + neg_inter_exp_logits.sum(1, keepdim=True))
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask_exc * log_prob).sum(1) / mask_exc.sum(1)
    mean_log_prob_pos_inter = (pos_inter_mask * log_prob_inter).sum(1) / (pos_inter_mask.sum(1))

    # Loss
    loss = (-mean_log_prob_pos-mean_log_prob_pos_inter ).mean()
    
    return loss

def supervised_contrastive_loss(features, labels, temperature):
    """
    Compute the supervised contrastive loss
    
    Args:
    - features: tensor of shape (batch_size, feature_dim)
    - labels: tensor of shape (batch_size,)
    - temperature: scalar value, temperature parameter
    
    Returns:
    - loss: scalar value, the supervised contrastive loss
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize feature vectors
    features = F.normalize(features, p=2, dim=1)
    
    # Compute pairwise cosine similarities
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create a mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Exclude self-contrast
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log_prob
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
    # Loss
    loss = -mean_log_prob_pos
    loss = loss.mean()
    
    return loss
    

class SlicedWasserstein(torch.nn.Module):
    def __init__(self, num_projections=128):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, x, y):
        assert x.shape == y.shape, "Input tensors must have the same shape"

        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), x.size(1), -1)
        y_flat = y.view(y.size(0), y.size(1), -1)

        # Generate random projections
        device = x.device
        projections = torch.randn(self.num_projections, x_flat.size(-1), device=device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)

        # Project the data
        x_proj = torch.matmul(x_flat, projections.t())
        y_proj = torch.matmul(y_flat, projections.t())

        # Sort projections
        x_proj_sort, _ = torch.sort(x_proj, dim=0)
        y_proj_sort, _ = torch.sort(y_proj, dim=0)

        # Compute Wasserstein distance
        w_dist = torch.mean(torch.abs(x_proj_sort - y_proj_sort))
        return w_dist

def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

class DSW(torch.nn.Module):
    def __init__(self, encoder, embedding_norm, num_projections, projnet, op_projnet,
                 p=2, max_iter=100, lam=1):
        super(DSW, self).__init__()
        self.encoder = encoder
        self.embedding_norm = embedding_norm
        self.num_projections = num_projections
        self.projnet = projnet
        self.op_projnet = op_projnet
        self.p = p
        self.max_iter = max_iter
        self.lam = lam

    def __cosine_distance_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

    @torch.enable_grad()
    @torch.inference_mode(mode=False)
    def distributional_sliced_wasserstein_distance(self, first_samples, second_samples):
        embedding_dim = first_samples.size(1)
        pro = rand_projections(embedding_dim, self.num_projections).to(first_samples)
        # sometimes first_samples and second_samples come from a context where torch.inference_mode is enabled so we need
        # to clone them first.
        first_samples_detach = first_samples.detach().clone()
        second_samples_detach = second_samples.detach().clone()
        # this tmp variable is for debugging purposes.
        tmp = []
        for _ in range(self.max_iter):
            projections = self.projnet(pro)
            cos = self.__cosine_distance_torch(projections, projections)
            reg = self.lam * cos
            encoded_projections = first_samples_detach.matmul(
                projections.transpose(0, 1))
            distribution_projections = second_samples_detach.matmul(
                projections.transpose(0, 1))
            wasserstein_distance = torch.abs(
                (
                        torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                        - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
                )
            )
            wasserstein_distance = torch.pow(
                torch.sum(torch.pow(wasserstein_distance + 1e-20, self.p), dim=1), 1.0 / self.p)
            wasserstein_distance = torch.pow(
                torch.pow(wasserstein_distance + 1e-20, self.p).mean(), 1.0 / self.p)
            loss = reg - wasserstein_distance
            self.op_projnet.zero_grad()
            loss.backward()
            self.op_projnet.step()
            tmp.append(wasserstein_distance.item())
        with torch.inference_mode():
            projections = self.projnet(pro)
        projections = projections.clone()
        encoded_projections = first_samples.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (
                    torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                    - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
            )
        )
        wasserstein_distance = torch.pow(
            torch.sum(torch.pow(wasserstein_distance, self.p), dim=1), 1.0 / self.p)
        wasserstein_distance = torch.pow(
            torch.pow(wasserstein_distance, self.p).mean(), 1.0 / self.p)
        return wasserstein_distance

    def forward(self, first_sample, second_sample):
        if self.encoder is None:
            data = second_sample
            data_fake = first_sample
        else:
            data = self.encoder(second_sample) / self.embedding_norm
            data_fake = self.encoder(first_sample) / self.embedding_norm
        # print(data.shape)
        # print(data_fake.shape)
        _dswd = self.distributional_sliced_wasserstein_distance(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1)
        )
        return _dswd

def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def ISEBSW(X, Y, L=256, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    wasserstein_distances =  wasserstein_distances.view(1,L)
    weights = torch.softmax(wasserstein_distances,dim=1)
    sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    return  torch.pow(sw,1./p)

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

def mmdfuse(
    X,
    Y,
    seed=None,
    alpha=0.05,
    kernels=("laplace", "gaussian"),
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=1,  # set permutations to 1
    return_p_val=False,
    device=None,
):
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    X = X.to(device)
    Y = Y.to(device)
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m = X.shape[0]
    n = Y.shape[0]
    assert n <= m
    assert n >= 2 and m >= 2
    assert 0 < alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and isinstance(number_bandwidths, int)
    assert number_permutations > 0 and isinstance(number_permutations, int)
    if isinstance(kernels, str):
        kernels = (kernels,)
    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Only 1 permutation + original order
    B = number_permutations
    idx = []
    for i in range(B + 1):
        if i < B:
            perm = torch.randperm(m + n, device=device)
            idx.append(perm)
        else:
            idx.append(torch.arange(m + n, device=device))
    idx = torch.stack(idx)

    v11 = torch.cat((torch.ones(m, device=device), -torch.ones(n, device=device)))
    V11i = v11.repeat(B + 1, 1)
    V11 = torch.gather(V11i, 1, idx).t()
    v10 = torch.cat((torch.ones(m, device=device), torch.zeros(n, device=device)))
    V10i = v10.repeat(B + 1, 1)
    V10 = torch.gather(V10i, 1, idx).t()
    v01 = torch.cat((torch.zeros(m, device=device), -torch.ones(n, device=device)))
    V01i = v01.repeat(B + 1, 1)
    V01 = torch.gather(V01i, 1, idx).t()

    N = number_bandwidths * number_kernels
    M = torch.zeros((N, B + 1), device=device)
    kernel_count = -1

    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            Z = torch.cat((X, Y))
            pairwise_matrix = torch_distances(Z, Z, l, matrix=True)

            def compute_bandwidths(distances, number_bandwidths):
                median = torch.median(distances)
                distances = distances + (distances == 0) * median
                dd, _ = torch.sort(distances)
                lambda_min = dd[int(torch.floor(torch.tensor(len(dd) * 0.05)).item())] / 2
                lambda_max = dd[int(torch.floor(torch.tensor(len(dd) * 0.95)).item())] * 2
                bandwidths = torch.linspace(lambda_min, lambda_max, number_bandwidths, device=device)
                return bandwidths

            indices = torch.triu_indices(pairwise_matrix.shape[0], pairwise_matrix.shape[0], 1, device=device)
            distances = pairwise_matrix[indices[0], indices[1]]
            bandwidths = compute_bandwidths(distances, number_bandwidths)

            # Detach bandwidths to prevent gradients flowing through them
            bandwidths = bandwidths.detach()

            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1
                for i in range(number_bandwidths):
                    bandwidth = bandwidths[i]
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    K = K - torch.diag_embed(torch.diagonal(K))
                    unscaled_std = torch.sqrt(torch.sum(K**2) + 1e-8)
                    # Compute MMD permuted values
                    M[kernel_count * number_bandwidths + i] = (
                        (
                            torch.sum(V10 * (K @ V10), 0)
                            * (n - m + 1)
                            * (n - 1)
                            / (m * (m - 1))
                            + torch.sum(V01 * (K @ V01), 0)
                            * (m - n + 1)
                            / m
                            + torch.sum(V11 * (K @ V11), 0)
                            * (n - 1)
                            / m
                        )
                        / unscaled_std
                        * torch.sqrt(torch.tensor(n * (n - 1), device=device, dtype=torch.float))
                    )

    all_statistics = torch.logsumexp(lambda_multiplier * M, dim=0) - torch.log(torch.tensor(N, device=device, dtype=torch.float))
    original_statistic = all_statistics[-1]  # (1,)
    
    # Return the original_statistic for backprop, as requested
    return original_statistic

def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """
    Compute kernel matrix for a given kernel and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return torch.exp(-(d**2) / 2)
    elif kernel == "laplace" and l == "l1":
        return torch.exp(-d * torch.sqrt(torch.tensor(2.0, device=d.device)))
    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (
        kernel == "matern_0.5_l2" and l == "l2"
    ):
        return torch.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (
        kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + torch.sqrt(torch.tensor(3.0, device=d.device)) * d) * torch.exp(-torch.sqrt(torch.tensor(3.0, device=d.device)) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (
        kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + torch.sqrt(torch.tensor(5.0, device=d.device)) * d + 5 / 3 * d**2) * torch.exp(-torch.sqrt(torch.tensor(5.0, device=d.device)) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (
        kernel == "matern_3.5_l2" and l == "l2"
    ):
        sqrt7 = torch.sqrt(torch.tensor(7.0, device=d.device))
        return (
            1 + sqrt7 * d + 2 * 7 / 5 * d**2 + 7 * sqrt7 / 3 / 5 * d**3
        ) * torch.exp(-sqrt7 * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (
        kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
            1
            + 3 * d
            + 3 * (6**2) / 28 * d**2
            + (6**3) / 84 * d**3
            + (6**4) / 1680 * d**4
        ) * torch.exp(-3 * d)
    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')

def torch_distances(X, Y, l, max_samples=None, matrix=False):
    """
    Compute pairwise distances between points in X and Y.
    
    Parameters:
    -----------
    X, Y: torch.Tensor
        Data matrices with shape (n_samples, n_features)
    l: str
        Type of distance: 'l1' for Manhattan, 'l2' for Euclidean
    max_samples: int or None
        Maximum number of samples to use
    matrix: bool
        If True, return the full distance matrix
        If False, return only the upper triangular part
        
    Returns:
    --------
    torch.Tensor: Pairwise distances
    """
    if max_samples is not None:
        X = X[:max_samples]
        Y = Y[:max_samples]
        
    n_x, n_y = X.shape[0], Y.shape[0]
    
    if l == "l1":
        # Manhattan distance
        # Compute |x_i - y_j| for all pairs
        X_expanded = X.unsqueeze(1)  # Shape: [n_x, 1, n_features]
        Y_expanded = Y.unsqueeze(0)  # Shape: [1, n_y, n_features]
        pairwise_dist = torch.sum(torch.abs(X_expanded - Y_expanded), dim=2)  # Shape: [n_x, n_y]
        
    elif l == "l2":
        # Euclidean distance
        # Compute ||x_i - y_j||_2 for all pairs
        X_norm = (X**2).sum(1).view(-1, 1)
        Y_norm = (Y**2).sum(1).view(1, -1)
        dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        # Ensure no negative values due to numerical errors
        dist = torch.clamp(dist, min=0.0)
        pairwise_dist = torch.sqrt(torch.clamp(dist, min=1e-8))
        
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    
    if matrix:
        return pairwise_dist
    else:
        # Return only upper triangular part for square matrices
        if n_x == n_y:
            indices = torch.triu_indices(n_x, n_y, 1, device=X.device)
            return pairwise_dist[indices[0], indices[1]]
        else:
            return pairwise_dist.view(-1)  # Flatten the matrix if not square
        
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Compute MMD loss between source and target distributions.
    
    Args:
        source (Tensor): Source samples (n_samples x n_features)
        target (Tensor): Target samples (n_samples x n_features)
        kernel_mul (float): Multiplier for kernel bandwidths
        kernel_num (int): Number of kernels to use
        fix_sigma (float): Fixed bandwidth if specified
        
    Returns:
        Tensor: MMD loss value
    """
    eps = 1e-8  # Numerical stability constant
    batch_size = source.size(0)
    
    # Combine samples
    total = torch.cat([source, target], dim=0)
    
    # Compute pairwise distances
    total = total.unsqueeze(0)
    L2_distance = ((total - total.permute(1, 0, 2))**2).sum(2)
    
    # Bandwidth selection
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.median(L2_distance.detach()) / (2 * torch.log(torch.tensor(L2_distance.size(0)) + 1))
        bandwidth = bandwidth.clamp(min=eps)
    
    # Multi-kernel bandwidths
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # Compute kernel values
    kernel_val = [torch.exp(-L2_distance / (b + eps)) for b in bandwidth_list]
    kernels = sum(kernel_val)
    
    # MMD computation
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    return torch.mean(XX + YY - XY - YX).clamp(min=0)
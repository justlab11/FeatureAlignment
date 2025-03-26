import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, group_labels, temperature):
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

def ISEBSW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    wasserstein_distances =  wasserstein_distances.view(1,L)
    weights = torch.softmax(wasserstein_distances,dim=1)
    sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    return  torch.pow(sw,1./p) + 1e-8
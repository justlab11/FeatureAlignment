import torch
import torch.nn.functional as F

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

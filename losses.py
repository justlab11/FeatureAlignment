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
    
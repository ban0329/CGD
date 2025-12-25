import torch
from scipy.stats import pearsonr



def fusion_pairwise(outputs: list[torch.Tensor]):
    """
    Perform pairwise correlation-based fusion for more than 2 models.

    Args:
        outputs (list[Tensor]): A list of model outputs, each with shape [batch_size, feature_size].

    Returns:
        Tensor: Fused vector with shape [batch_size, feature_size].
    """
    num_models = len(outputs)
    batch_size, feature_size = outputs[0].shape
    device = outputs[0].device
    # Initialize fused vector
    fused_vector = torch.zeros(batch_size, feature_size)

    for i in range(batch_size):
        # Extract feature vectors for the current batch
        vectors = [outputs[m][i, :].cpu().squeeze() for m in range(num_models)]

        # Initialize weights and fused vector
        weights = torch.zeros(num_models)
        vec_fusion = torch.zeros(feature_size)

        # Compute pairwise correlations and weights for each model
        for m1 in range(num_models):
            pairwise_corr_sum = 0
            for m2 in range(num_models):
                if m1 != m2:
                    corr, _ = pearsonr(vectors[m1], vectors[m2])
                    pairwise_corr_sum += abs(corr)

            # Assign weight inversely proportional to average pairwise correlation
            weights[m1] = 1 - (pairwise_corr_sum / (num_models - 1))  # Exclude self-correlation

        # Normalize weights to sum to 1
        #weights /= weights.sum()

        # Compute fused vector as weighted sum of vectors
        for m, weight in enumerate(weights):
            vec_fusion += weight * vectors[m]

        fused_vector[i, :] = vec_fusion

    return fused_vector.to(device)

def fusion(outputs: torch.Tensor):
    """
    Perform correlation-based fusion for two models using Pearson correlation.

    This method fuses two sets of feature representations based on their
    pairwise Pearson correlation. The intuition is that when the two models
    produce highly correlated features, the fusion relies more on the
    correlated model (model2), while decorrelated features receive higher
    weight from the less correlated model (model1).

    Args:
        outputs (Tensor): A list containing two tensors, each of shape
                          [batch_size, feature_size].

    Returns:
        Tensor: The fused feature tensor of shape [batch_size, feature_size].
    """
    output1 = outputs[0]
    output2 = outputs[1]
    batch_size, feature_size = output1.shape

    # Initialize fused feature tensor
    fused_vector = torch.zeros(batch_size, feature_size)
    correlations = []

    for i in range(batch_size):
        device = output1.device

        # Extract feature vectors for the current sample
        vec1 = output1[i, :].cpu().squeeze()
        vec2 = output2[i, :].cpu().squeeze()

        # Compute Pearson correlation between the two model outputs
        corr, _ = pearsonr(vec1, vec2)

        # Assign adaptive weights based on correlation strength
        # Less correlated features receive higher weight from model1
        weight_model1 = 1 - abs(corr)
        weight_model2 = abs(corr)

        # Fuse features using correlation-based weighting
        vec3 = weight_model1 * vec1 + weight_model2 * vec2

        fused_vector[i, :] = vec3

    # Move fused result back to the original device
    fused_vector = fused_vector.to(device)
    return fused_vector

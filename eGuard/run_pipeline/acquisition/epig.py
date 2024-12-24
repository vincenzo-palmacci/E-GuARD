import torch
import logging
import math

from numpy import stack

# Functions originally implemented by Bickford Smith et al. (2023) https://github.com/fbickfordsmith/epig
def conditional_epig_from_probs(
    probs_pool: torch.Tensor, 
    probs_targ: torch.Tensor,
    batch_size: int = 100
) -> torch.Tensor:
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        batch_size: int, size of the batch to process at a time

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Get sizes
    N_p, K, Cl = probs_pool.size()
    N_t = probs_targ.size(0)

    # Prepare tensors
    scores = torch.zeros(N_p, N_t)

    # Process in batches to save memory
    for i in range(0, N_p, batch_size):
        for j in range(0, N_t, batch_size):
            # Get the batch
            probs_pool_batch = probs_pool[i:i + batch_size]
            probs_targ_batch = probs_targ[j:j + batch_size]

            # Estimate the joint predictive distribution.
            probs_pool_batch = probs_pool_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_targ_batch = probs_targ_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_pool_batch = probs_pool_batch[:, :, None, :, None]  # [K, batch_size, 1, Cl, 1]
            probs_targ_batch = probs_targ_batch[:, None, :, None, :]  # [K, 1, batch_size, 1, Cl]
            probs_pool_targ_joint = probs_pool_batch * probs_targ_batch
            probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)

            # Estimate the marginal predictive distributions.
            probs_pool_batch = torch.mean(probs_pool_batch, dim=0)
            probs_targ_batch = torch.mean(probs_targ_batch, dim=0)

            # Estimate the product of the marginal predictive distributions.
            probs_pool_targ_indep = probs_pool_batch * probs_targ_batch

            # Estimate the conditional expected predictive information gain for each pair of examples.
            # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
            nonzero_joint = probs_pool_targ_joint > 0
            log_term = torch.clone(probs_pool_targ_joint)
            log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])
            log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])
            score_batch = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))
            
            # Store the results
            scores[i:i + batch_size, j:j + batch_size] = score_batch

    return scores  # [N_p, N_t]


def check(
    scores: torch.Tensor, max_value: float = math.inf, epsilon: float = 1e-6, score_type: str = ""
) -> torch.Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        
        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")
    
    return scores

def epig_from_conditional_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]

def epig_from_probs(probs_pool: torch.Tensor, probs_targ: torch.Tensor) -> torch.Tensor:
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]

def get_prob_distribution(model, x):
    prob_dist = [estimator.predict_proba(x) for estimator in model.estimators_]
    prob_dist = stack(prob_dist, axis = 1)
    prob_dist = torch.tensor(prob_dist)
    return prob_dist
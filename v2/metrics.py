import numpy as np
import torch
import warnings
from typing import Union


def pearsonr_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average Pearson correlation.
    To do that, it takes the i-th column of each matrix and computes the Pearson correlation between them.
    It finally returns the average of all the Pearson correlations computed.

    Args:
        gt_mat (np.array): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (np.array): Predicted matrix of shape (n_observations, n_variables).
    
    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
    """
    # Center both matrices by subtracting the mean of each column
    centered_gt_mat = gt_mat - gt_mat.mean(dim=0)
    centered_pred_mat = pred_mat - pred_mat.mean(dim=0)

    # Compute pearson correlation with cosine similarity
    pcc = torch.nn.functional.cosine_similarity(centered_gt_mat, centered_pred_mat, dim=0)

    # Compute mean pearson correlation
    mean_pcc = pcc.mean().item()

    return mean_pcc

def pearsonr_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the gene dimension. It is computing the correlation between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).

    Returns:
        float: Mean Pearson correlation computed by averaging the Pearson correlation for each gene.
    """
    return pearsonr_cols(gt_mat=gt_mat, pred_mat=pred_mat)

def pearsonr_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the patch dimension. It is computing the correlation the between true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).

    Returns:
        float: Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
    """
    # Transpose matrices and apply pearsonr_torch_cols 
    return pearsonr_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T)

def r2_score_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average R2 score.
    To do that, it takes the i-th column of each matrix and computes the R2 score between them.
    It finally returns the average of all the R2 scores computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each column in the matrices.
    """

    # Compute the column means of the ground truth
    gt_col_means = gt_mat.mean(dim=0)
    
    # Compute the total sum of squares
    total_sum_squares = torch.sum(torch.square(gt_mat - gt_col_means), dim=0)

    # Compute the residual sum of squares
    residual_sum_squares = torch.sum(torch.square(gt_mat - pred_mat), dim=0)

    # Compute the R2 score for each column
    r2_scores = 1 - (residual_sum_squares / total_sum_squares)

    # Compute the mean R2 score
    mean_r2_score = r2_scores.mean().item()

    return mean_r2_score

def r2_score_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the gene dimension. It is computing the R2 score between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each gene.
    """
    return r2_score_cols(gt_mat=gt_mat, pred_mat=pred_mat)

def r2_score_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor) -> float:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the patch dimension. It is computing the R2 score between the true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each patch.
    """
    # Transpose matrices and apply r2_score_torch_cols
    return r2_score_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T)

def get_metrics(gt_mat: Union[np.array, torch.Tensor] , pred_mat: Union[np.array, torch.Tensor]) -> dict:
    """
    This function receives 2 matrices of shapes (n_samples, n_genes) and computes the following metrics:
    
        - Pearson correlation (gene-wise) [PCC-Gene]
        - Pearson correlation (patch-wise) [PCC-Patch]
        - r2 score (gene-wise) [R2-Gene]
        - r2 score (patch-wise) [R2-Patch]
        - Mean squared error [MSE]
        - Mean absolute error [MAE]

    Args:
        gt_mat (Union[np.array, torch.Tensor]): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (Union[np.array, torch.Tensor]): Predicted matrix of shape (n_samples, n_genes).

    Returns:
        dict: Dictionary containing the metrics computed. The keys are: ['PCC-Gene', 'PCC-Patch', 'R2-Gene', 'R2-Patch', 'MSE', 'MAE']
    """

    # Assert that both matrices have the same shape
    assert gt_mat.shape == pred_mat.shape, "Both matrices must have the same shape."

    # If input are numpy arrays, convert them to torch tensors
    if isinstance(gt_mat, np.ndarray):
        gt_mat = torch.from_numpy(gt_mat)
    if isinstance(pred_mat, np.ndarray):
        pred_mat = torch.from_numpy(pred_mat)

    # Get boolean indicating constant columns in predicted matrix 
    # NOTE: A constant gene prediction will mess with the pearson correlation
    constant_cols = torch.all(pred_mat == pred_mat[[0],:], axis = 0)
    # Get boolean indicating if there are any constant columns
    any_constant_cols = torch.any(constant_cols)

    # Get boolean indicating constant rows in predicted matrix
    # NOTE: A constant patch prediction will mess with the pearson correlation
    constant_rows = torch.all(pred_mat == pred_mat[:,[0]], axis = 1)
    # Get boolean indicating if there are any constant rows
    any_constant_rows = torch.any(constant_rows)

    # If there are any constant columns, set the pcc_g and r2_g to None
    if any_constant_cols:
        pcc_g = None
        warnings.warn("There are constant columns in the predicted matrix. This means a gene is being predicted as constant. The Pearson correlation (gene-wise) will be set to None.")
    else:
        # Compute Pearson correlation (gene-wise)
        pcc_g = pearsonr_gene(gt_mat, pred_mat)
    
    # If there are any constant rows, set the pcc_p and r2_p to None
    if any_constant_rows:
        pcc_p = None
        warnings.warn("There are constant rows in the predicted matrix. This means a patch is being predicted as constant. The Pearson correlation (patch-wise) will be set to None.")
    else:
        # Compute Pearson correlation (patch-wise)
        pcc_p = pearsonr_patch(gt_mat, pred_mat)
        

    # Compute r2 score (gene-wise)
    r2_g = r2_score_gene(gt_mat, pred_mat)
    # Compute r2 score (patch-wise)
    r2_p = r2_score_patch(gt_mat, pred_mat)
    # Compute mean squared error
    mse = torch.nn.functional.mse_loss(gt_mat, pred_mat, reduction='mean').item()
    # Compute mean absolute error
    mae = torch.nn.functional.l1_loss(gt_mat, pred_mat, reduction='mean').item()

    # Create dictionary with the metrics computed
    metrics_dict = {
        'PCC-Gene': pcc_g,
        'PCC-Patch': pcc_p,
        'R2-Gene': r2_g,
        'R2-Patch': r2_p,
        'MSE': mse,
        'MAE': mae,
        'Global': pcc_g+pcc_p+r2_g+r2_p-mse-mae
    }

    return metrics_dict

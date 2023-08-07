"""
Code originally from Google's Chirp project implementation of NOTELA:
https://github.com/google-research/chirp/blob/main/chirp/projects/sfda/methods/notela.py
"""
import numpy as np
import torch
import scipy
from pyha_analyzer.models.timm_model import TimmModel

def compute_nearest_neighbors(batch_feature: np.ndarray,
                              dataset_feature: np.ndarray,
                              knn: int,
                              memory_efficient_computation: bool = False) -> np.ndarray:
    """ Algorithm to compute the nearest neighbors between two sets of features. """
    batch_shape = batch_feature.shape
    dataset_shape = dataset_feature.shape

    if batch_feature.ndim != dataset_feature.ndim or (
        batch_shape[-1] != dataset_shape[-1]
    ):

        raise ValueError(
            "Batch features and dataset features' shapes are not consistent."
            f"(batch_feature: {batch_shape} and dataset_feature: {dataset_shape})"
        )

    neighbors = min(dataset_shape[0], knn)
    if memory_efficient_computation:
        # We loop over samples in the current batch to avoid storing a
        # batch_size x dataset_size float array. That slows down computation, but
        # reduces memory footprint, which becomes the bottleneck for large
        # datasets.
        col_indices = []
        for sample_feature in batch_feature:
            pairwise_distances = scipy.spatial.distance.cdist(
                np.expand_dims(sample_feature, 0), dataset_feature
            )  # [1, dataset_size]
            col_indices.append(
                torch.topk(-pairwise_distances, neighbors)[1][:, 1:]
            )  # [1, neighbors-1]
        col_indices = torch.stack(col_indices)
    else:

        pairwise_distances = scipy.spatial.distance.cdist(
            batch_feature, dataset_feature
        )  # [batch_size, dataset_size]
        col_indices = torch.topk(-pairwise_distances, neighbors)[1][
            :, 1:
        ]  # [batch_size, neighbors-1]
    col_indices = col_indices.flatten()  # [batch_size * neighbors-1]
    row_indices = np.repeat(
        np.arange(batch_shape[0]), neighbors - 1
    )  # [0, ..., 0, 1, ...]
    nn_matrix = np.zeros((batch_shape[0], dataset_shape[0]), dtype=np.uint8)
    nn_matrix = nn_matrix[row_indices, col_indices].set(1)
    return nn_matrix

def teacher_step(
        batch_prob: np.ndarray,
        dataset_prob: np.ndarray,
        nn_matrix: np.ndarray,
        lambda_: float,
        alpha: float = 1.0,
        normalize_pseudo_labels: bool = True,
        eps: float = 1e-8
        ) -> np.ndarray:
    """Computes the pseudo-labels (teacher-step) following Eq.(3) in the paper.

    Args:
      batch_proba: The model's probabilities on the current batch of data.
        Expected shape [batch_size, proba_dim]
      dataset_proba: The model's probabilities on the rest of the dataset.
        Expected shape [dataset_size, proba_dim]
      nn_matrix: The affinity between the points in the current batch
        (associated to `batch_proba`) and the remaining of the points
        (associated to `dataset_proba`), of shape [batch_size, dataset_size].
        Specifically, position [i,j] informs if point j belongs to i's
        nearest-neighbors.
      lambda_: Weight controlling the Laplacian regularization.
      alpha: Weight controlling the Softness regularization
      normalize_pseudo_labels: Whether to normalize pseudo-labels to turn them
        into valid probability distributions. This option should be kept to
        True, and only be used for experimental purposes.
      eps: For numerical stability.

    Returns:
      The soft pseudo-labels for the current batch of data, shape
        [batch_size, proba_dim]
    """
    denominator = nn_matrix.sum(axis=-1, keepdims=True)

    # In the limit where alpha goes to zero, we can rewrite the expression as
    #
    #     pseudo_label = [batch_proba * jnp.exp(lambda_ * ...)] ** (1 / alpha)
    #
    # and see that the normalized pseudo-label probabilities take value 1 if
    # they have the maximum value for the expression above over the class axis
    # and zero otherwise.
    if alpha == 0 and normalize_pseudo_labels:
        pseudo_label = batch_prob * np.exp(
            lambda_
            * (nn_matrix @ dataset_prob)
            / (denominator + eps)  # [*, batch_size, proba_dim]
        )
        pseudo_label = (
            pseudo_label == pseudo_label.max(axis=-1, keepdims=True)
        ).astype(np.float32)
        # If more than one class is maximally probable, we need to renormalize the
        # distribution to be uniform over the maximally-probable classes.
        pseudo_label /= pseudo_label.sum(axis=-1, keepdims=True)
    else:
        pseudo_label = batch_prob ** (1 / alpha) * np.exp(
            (lambda_ / alpha) * (nn_matrix @ dataset_prob) / (denominator + eps)
        )  # [*, batch_size, proba_dim]
        if normalize_pseudo_labels:
            pseudo_label /= pseudo_label.sum(axis=-1, keepdims=True) + eps
    return pseudo_label

if __name__=="__main__":
    model = TimmModel(10)
    image = torch.rand((1, 3, 100, 100))
    features = model.get_features(image)
    print(f"{features=}")

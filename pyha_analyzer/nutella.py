"""
Code originally from Google's Chirp project implementation of NOTELA:
https://github.com/google-research/chirp/blob/main/chirp/projects/sfda/methods/notela.py
"""
import numpy as np
import scipy
import torch

from pyha_analyzer import pseudolabel, dataset, config
from pyha_analyzer.models.timm_model import TimmModel

cfg = config.cfg

def compute_nearest_neighbors(
        batch_feature: np.ndarray,
        dataset_feature: np.ndarray,
        knn: int,
        memory_efficient_computation: bool = False
    ) -> np.ndarray:
    """
    Compute batch_feature's nearest-neighbors among dataset_feature.
    
        Args:
            batch_feature: The features for the provided batch of data, 
                shape [batch_size, feature_dim]
            dataset_feature: The features for the whole dataset, 
                shape [dataset_size, feature_dim]
            knn: The number of nearest-neighbors to use.
            memory_efficient_computation: Whether to make computation memory
                efficient. This option trades speed for memory footprint by looping over
                samples in the batch instead of fully vectorizing nearest-neighbor
                computation. For large datasets, memory usage can be a bottleneck, which
                is why we set this option to True by default.
            
        Returns:
            The batch's nearest-neighbors affinity matrix of shape [batch_size, dataset_size], 
            where position (i, j) indicates whether dataset_feature[j] 
            belongs to batch_feature[i]'s nearest-neighbors.
            
        Raises:
            ValueError: If batch_feature and dataset_feature don't have the same
            number of dimensions, or if their feature dimension don't match.
    """

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

def get_dataset_info(model, dl):
    indices = data = predictions = features = []
    print(f"{len(dl)}")
    for _, (mels, _, index) in enumerate(dl):
        dataset.append(mels)
        predictions.append(model(mels))
        predictions.append(model.get_features(mels))
        indices.append(index)
    print(f"{len(indices)}")
    print(f"{len(data)}")
    print(f"{len(predictions)}")
    print(f"{len(features)}")
    return [torch.cat(l) for l in (indices, data, predictions, features)]

def regularized_pseudolabels(model, dataset):
    dataset_features = model.get_features(dataset)
    # Might need to rewrite iteratively for less memory
    dataset_predictions = model(dataset)
    nn_matrix = compute_nearest_neighbors(
            batch_features, dataset_features, knn=cfg.notela_knn
    ) # [dataset_size, dataset_size]
    pseudo_labels = teacher_step(
            dataset_predictions, dataset_predictions,
            nn_matrix,
            lambda_ = cfg.notela_lambda
    ) # [dataset_size, num_classes]
    return pseudo_labels #TODO: Convert to one hot

def finetune(model):
    """
    Fine tune on pseudo labels
    """
    pseudo_df, train_dl, valid_dl, infer_dl = pseudolabel.pseudo_label_data(model)

    logger.info("Finetuning on pseudo labels...")
    train_process = TrainProcess(model, train_dl, valid_dl, infer_dl)
    train_process.valid()
    train_process.inference_valid()
    for _ in range(cfg.epochs):
        dataset, indices, predictions, features = get_dataset_info(train_dl)
        pseudo_labels = regularized_pseudolabels(TrainProcess.model, dataset)
        #train_process.update_dataset_predictions(indices, pseudolabel) #TBD
        train_process.run_epoch()
        train_process.valid()
        train_process.inference_valid()

if __name__=="__main__":
    model = TimmModel(3).to(cfg.device)
    finetune(model)

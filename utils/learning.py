import random
import math
import numpy as np
import torch
from torch import nn
import os
import torch.utils.data as data
from tqdm import tqdm


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=40, n_epochs=1000, lr_input=0.001):
    """
    Decay the learning rate with a half-cycle cosine after warmup.

    Parameters:
    - optimizer: The optimizer to adjust the learning rate for.
    - epoch: The current epoch number.
    - warmup_epochs: The number of warmup epochs.
    - n_epochs: The total number of epochs.
    - lr_input: The initial learning rate.

    Returns:
    - lr: The adjusted learning rate.
    """
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def cast_label_to_one_hot_and_prototype(y_labels_batch, n_class, return_prototype=False):
    """
    Convert labels to one-hot encoding and optionally return the prototype.

    Parameters:
    - y_labels_batch: A vector of length batch_size.
    - n_class: The number of classes.
    - return_prototype: Whether to return the prototype.

    Returns:
    - y_one_hot_batch: The one-hot encoded labels.
    - y_logits_batch (optional): The prototype logits if return_prototype is True.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=n_class).float()
    if return_prototype:
        label_min, label_max = [0.001, 0.999]
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


def init_fn(worker_id):
    """
    Initialize the random seed for data loader workers.

    Parameters:
    - worker_id: The worker ID.
    """
    np.random.seed(77 + worker_id)

def prepare_2_fp_x(fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400):
    """
    Prepare feature embeddings for weak and strong augmentations.

    Parameters:
    - fp_encoder: The feature extractor.
    - dataset: The dataset to extract features from.
    - save_dir: The directory to save the embeddings.
    - device: The device to perform computation on.
    - fp_dim: The dimension of the feature embeddings.
    - batch_size: The batch size for data loading.

    Returns:
    - fp_embed_all_weak: The weakly augmented feature embeddings.
    - fp_embed_all_strong: The strongly augmented feature embeddings.
    """
    # Check if precomputed features already exist
    if save_dir is not None:
        if os.path.exists(save_dir + '_weak.npy') and os.path.exists(save_dir + '_strong.npy'):
            fp_embed_all_weak = torch.tensor(np.load(save_dir + '_weak.npy'))
            fp_embed_all_strong = torch.tensor(np.load(save_dir + '_strong.npy'))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all_weak.cpu(), fp_embed_all_strong.cpu()

    # Initialize two sets of feature spaces for weak and strong augmentations
    fp_embed_all_weak = torch.zeros([len(dataset), fp_dim], device=device)
    fp_embed_all_strong = torch.zeros([len(dataset), fp_dim], device=device)

    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing embeddings fp(x)', ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch_weak, x_batch_strong, _, data_indices] = data_batch[:4]
                temp_weak = fp_encoder(x_batch_weak.to(device))
                temp_strong = fp_encoder(x_batch_strong.to(device))
                data_indices = data_indices.to(device)
                fp_embed_all_weak[data_indices] = temp_weak
                fp_embed_all_strong[data_indices] = temp_strong

    # Save the computed features (if save directory is specified)
    if save_dir is not None:
        np.save(save_dir + '_weak.npy', fp_embed_all_weak.cpu().numpy())
        np.save(save_dir + '_strong.npy', fp_embed_all_strong.cpu().numpy())

    return fp_embed_all_weak.cpu(), fp_embed_all_strong.cpu()


def prepare_fp_x(fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400):
    """
    Prepare feature embeddings for the dataset.

    Parameters:
    - fp_encoder: The feature extractor.
    - dataset: The dataset to extract features from.
    - save_dir: The directory to save the embeddings.
    - device: The device to perform computation on.
    - fp_dim: The dimension of the feature embeddings.
    - batch_size: The batch size for data loading.

    Returns:
    - fp_embed_all: The feature embeddings.
    """
    if save_dir is not None:
        if os.path.exists(save_dir):
            fp_embed_all = torch.tensor(np.load(save_dir))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all.cpu()

    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        fp_embed_all = torch.zeros([len(dataset), fp_dim]).to(device)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing embeddings fp(x)', ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch, _, data_indices] = data_batch[:3]
                temp = fp_encoder(x_batch.to(device))
                fp_embed_all[data_indices, :] = temp

        if save_dir is not None:
            np.save(save_dir, fp_embed_all.cpu())

    return fp_embed_all.cpu()


def cnt_agree(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions for the specified values of k.

    Parameters:
    - output: The model output.
    - target: The ground truth labels.
    - topk: The list of top k values.

    Returns:
    - The number of correct predictions.
    """
    maxk = min(max(topk), output.size()[1])

    output = torch.softmax(-(output - 1) ** 2, dim=-1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return torch.sum(correct).item()



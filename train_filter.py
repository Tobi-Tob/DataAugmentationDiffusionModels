from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.road_sign import RoadSignDataset
from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights

import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

DATASETS = {
    "spurge": SpurgeDataset,
    "coco": COCODataset,
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset,
    "road_sign": RoadSignDataset
}


def train_filter(examples_per_class,
                 seed: int,
                 dataset: str,
                 image_size: int,
                 iterations_per_epoch: int = 200,
                 max_epochs: int = 100,
                 batch_size: int = 32,
                 model_dir: str = "models",
                 lr: float = 1e-4,
                 weight_decay: float = 1e-2,
                 use_randaugment: bool = True,
                 early_stopping_threshold: int = 10):  # Number of epochs without improvement trigger early stopping
    """
    Trains a classifier on the training data using a weighted sampler to address imbalances in class distribution
    and saves the model version with the best validation loss.
    This saved model is intended for later use in filtering synthetic images.

    Filtered Images trained on whole dataset with threshold 0.25:
    {0: 30, 1: 6, 2: 30, 3: 5, 5: 7, 7: 29, 8: 9, 9: 7, 10: 1, 12: 24, 13: 4, 14: 2, 15: 10, 16: 4, 17: 14, 19: 2, 20: 2, 21: 1, 24: 13, 25: 5, 26: 16, 27: 4, 28: 24, 30: 1, 31: 5, 32: 19, 34: 17, 35: 10, 36: 7, 37: 2, 39: 28, 40: 2, 41: 6, 42: 47, 43: 10, 44: 18, 45: 39, 47: 14, 48: 17, 49: 9, 50: 17, 51: 20, 52: 12, 53: 1, 54: 6, 55: 4, 56: 43, 57: 43, 58: 28, 59: 12, 60: 32, 61: 8, 62: 43, 63: 23, 64: 6, 65: 6, 66: 9, 68: 27, 69: 8, 70: 33, 71: 28, 72: 5, 73: 21, 74: 6, 75: 20, 76: 9, 77: 2, 78: 44, 79: 29}
    {0: 33, 2: 32, 3: 1, 4: 1, 5: 3, 7: 17, 8: 3, 9: 3, 10: 12, 11: 1, 12: 19, 13: 18, 14: 12, 15: 12, 17: 1, 20: 3, 24: 12, 25: 7, 26: 23, 27: 19, 28: 15, 29: 7, 30: 6, 31: 11, 32: 23, 33: 4, 34: 33, 35: 4, 36: 5, 38: 3, 39: 26, 40: 8, 41: 11, 42: 50, 43: 25, 44: 21, 45: 41, 46: 1, 47: 14, 48: 4, 49: 3, 50: 28, 51: 22, 52: 5, 55: 9, 56: 60, 57: 22, 58: 13, 59: 14, 60: 44, 61: 15, 62: 3, 63: 18, 64: 7, 65: 2, 66: 1, 67: 5, 68: 34, 69: 9, 70: 49, 71: 35, 72: 14, 73: 14, 74: 9, 75: 7, 76: 1, 77: 9, 78: 49, 79: 24}

    Filtered Images trained on 8 examples per class with threshold 0.20:
    {0: 14, 1: 10, 2: 11, 3: 7, 5: 4, 6: 1, 7: 5, 8: 1, 9: 5, 11: 1, 12: 27, 13: 5, 14: 2, 15: 7, 16: 1, 17: 2, 20: 2, 21: 1, 24: 3, 25: 7, 26: 1, 27: 2, 28: 6, 29: 7, 30: 5, 31: 2, 32: 18, 34: 23, 35: 4, 36: 9, 37: 4, 39: 10, 40: 5, 41: 9, 42: 30, 43: 3, 44: 7, 45: 29, 47: 12, 48: 6, 49: 4, 50: 10, 51: 2, 52: 1, 54: 6, 55: 2, 56: 49, 57: 45, 58: 15, 59: 14, 60: 34, 61: 8, 62: 31, 63: 14, 64: 4, 65: 4, 66: 8, 67: 3, 68: 14, 69: 13, 70: 30, 71: 20, 72: 15, 73: 9, 74: 2, 75: 8, 76: 5, 77: 2, 78: 39, 79: 35}

    Filtered Images trained on 8 examples per class with threshold 0.25:
    {0: 15, 1: 10, 2: 14, 3: 7, 5: 5, 6: 1, 7: 7, 8: 3, 9: 8, 10: 2, 11: 1, 12: 35, 13: 7, 14: 6, 15: 7, 16: 1, 17: 2, 19: 1, 20: 3, 21: 1, 24: 9, 25: 11, 26: 4, 27: 6, 28: 8, 29: 11, 30: 9, 31: 8, 32: 25, 33: 1, 34: 27, 35: 5, 36: 16, 37: 5, 39: 12, 40: 6, 41: 13, 42: 34, 43: 6, 44: 11, 45: 35, 46: 1, 47: 16, 48: 14, 49: 5, 50: 13, 51: 7, 52: 1, 54: 10, 55: 3, 56: 55, 57: 50, 58: 22, 59: 22, 60: 43, 61: 11, 62: 42, 63: 20, 64: 11, 65: 6, 66: 10, 67: 6, 68: 22, 69: 18, 70: 36, 71: 27, 72: 23, 73: 14, 74: 3, 75: 13, 76: 10, 77: 2, 78: 45, 79: 42}
    {0: 52, 1: 20, 2: 32, 3: 4, 4: 4, 6: 2, 7: 22, 8: 6, 9: 12, 10: 18, 11: 4, 12: 54, 13: 42, 14: 23, 15: 25, 16: 3, 17: 8, 19: 21, 20: 8, 21: 1, 22: 1, 24: 16, 25: 18, 26: 12, 27: 12, 28: 47, 29: 21, 30: 6, 31: 21, 32: 24, 33: 6, 34: 45, 35: 6, 36: 38, 37: 1, 38: 7, 39: 18, 40: 16, 41: 20, 42: 50, 43: 30, 44: 26, 45: 52, 46: 8, 47: 34, 48: 8, 49: 3, 50: 17, 51: 29, 52: 3, 53: 1, 54: 5, 55: 21, 56: 66, 57: 41, 58: 23, 59: 27, 60: 39, 61: 45, 62: 14, 63: 23, 64: 13, 65: 20, 66: 3, 67: 14, 68: 49, 69: 25, 70: 57, 71: 34, 72: 41, 73: 15, 74: 10, 75: 27, 76: 11, 77: 15, 78: 59, 79: 46}

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train",
        examples_per_class=examples_per_class,
        synthetic_probability=0,
        use_randaugment=use_randaugment,
        seed=seed,
        image_size=(image_size, image_size)
    )

    # Calculate class weights based on the inverse of class frequencies. Assign weight to each sample in the dataset
    # based on the class distribution, so that each class has an equal contribution to the overall loss.
    # If class_count is 0 set the corresponding entry in class_weights to 0 too.
    class_weights = np.where(train_dataset.class_counts == 0, 0, 1.0 / train_dataset.class_counts)
    weights = [class_weights[label] for label in train_dataset.all_labels]

    weighted_train_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=weighted_train_sampler, num_workers=4)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size))

    # TL: RuntimeWarning divide by zero can happen, everything will work as it should,
    # but this means that some classes are not present in the validation dataset.
    class_weights = np.where(val_dataset.class_counts == 0, 0, 1.0 / val_dataset.class_counts)
    weights = [class_weights[label] for label in val_dataset.all_labels]

    weighted_val_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=weighted_val_sampler, num_workers=4)

    filter_model = ClassificationFilterModel(
        train_dataset.num_classes
    ).cuda()

    optim = torch.optim.Adam(filter_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_validation_loss = np.inf
    corresponding_validation_accuracy = 0
    best_filter_model = None
    no_improvement_counter = 0

    records = []

    progress_bar = tqdm(range(max_epochs), desc="Training Filter")
    for epoch in progress_bar:

        filter_model.train()

        epoch_loss = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = filter_model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1:label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():

                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy().mean()
        training_accuracy = training_accuracy.cpu().numpy().mean()

        filter_model.eval()

        epoch_loss = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = filter_model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy().mean()
        validation_accuracy = validation_accuracy.cpu().numpy().mean()

        progress_bar.set_postfix({'train_loss': training_loss,
                                  'val_loss': validation_loss,
                                  'val_accuracy': validation_accuracy})

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=training_loss,
            metric="Loss",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=validation_loss,
            metric="Loss",
            split="Validation"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=training_accuracy,
            metric="Accuracy",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=validation_accuracy,
            metric="Accuracy",
            split="Validation"
        ))

        # Check if the current epoch has the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            corresponding_validation_accuracy = validation_accuracy
            best_filter_model = filter_model.state_dict()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Check for early stopping
        if no_improvement_counter >= early_stopping_threshold:
            print(
                f"No improvement in validation accuracy for {early_stopping_threshold} epochs. Stopping training.")
            break

    # Safe the logs
    log_path = f"logs/train_filter_{seed}_{epoch + 1}x{iterations_per_epoch}.csv"
    pd.DataFrame.from_records(records).to_csv(log_path)

    # Load the best model
    filter_model.load_state_dict(best_filter_model).cude()

    """
    Copyright (c) 2017 Geoff Pleiss
    https://github.com/gpleiss/temperature_scaling
    Tune the temperature of the model using the validation set.
    We're going to set it to optimize NLL.
    """
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()

    # Collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for image, label in val_dataloader:
            image = image.cuda()

            logits = filter_model(image)

            logits_list.append(logits)
            labels_list.append(label)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Start temperature: %.3f' % filter_model.temperature.item())
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: Optimize the temperature w.r.t. NLL
    temp_optimizer = torch.optim.LBFGS([filter_model.temperature], lr=0.01, max_iter=10)

    def eval():
        temp_optimizer.zero_grad()
        loss = nll_criterion(filter_model.temperature_scale(logits), labels)
        loss.backward()
        return loss

    temp_optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(filter_model.temperature_scale(logits), labels).item()
    after_temperature_ece = ece_criterion(filter_model.temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % filter_model.temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))


    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ClassificationFilterModel.pth"
    torch.save(filter_model, model_path)

    print(f"Model saved to {model_path} - Validation loss {best_validation_loss} - Validation accuracy "
          f"{corresponding_validation_accuracy} - Training results saved to {log_path}")


class ClassificationFilterModel(nn.Module):

    def __init__(self, num_classes: int):
        super(ClassificationFilterModel, self).__init__()

        self.image_processor = None
        self.temperature = nn.Parameter(torch.ones(1) * 1)

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, image):
        x = image
        with torch.no_grad():
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)

        return self.out(x)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Filter")

    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["spurge", "coco", "pascal", "flowers", "road_sign"])
    parser.add_argument("--examples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    args = parser.parse_args()

    train_filter(examples_per_class=args.examples_per_class,
                 seed=args.seed,
                 dataset=args.dataset,
                 image_size=256,
                 weight_decay=args.weight_decay)

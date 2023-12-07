from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
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
    "flowers": Flowers102Dataset
}


def train_filter(examples_per_class,
                 seed: int = 0,
                 dataset: str = "coco",
                 iterations_per_epoch: int = 200,
                 max_epochs: int = 100,
                 batch_size: int = 32,
                 image_size: int = 256,
                 model_dir: str = "models",
                 lr: float = 1e-4,
                 weight_decay: float = 1e-3,
                 early_stopping_threshold: int = 10):  # Number of epochs without improvement trigger early stopping
    """
    Trains a classifier on the training data using a weighted sampler to address imbalances in class distribution
    and saves the model version with the best validation loss.
    This saved model is intended for later use in filtering synthetic images.

    Filtered Images trained on whole dataset with threshold 0.25:
    {0: 30, 1: 6, 2: 30, 3: 5, 5: 7, 7: 29, 8: 9, 9: 7, 10: 1, 12: 24, 13: 4, 14: 2, 15: 10, 16: 4, 17: 14, 19: 2, 20: 2, 21: 1, 24: 13, 25: 5, 26: 16, 27: 4, 28: 24, 30: 1, 31: 5, 32: 19, 34: 17, 35: 10, 36: 7, 37: 2, 39: 28, 40: 2, 41: 6, 42: 47, 43: 10, 44: 18, 45: 39, 47: 14, 48: 17, 49: 9, 50: 17, 51: 20, 52: 12, 53: 1, 54: 6, 55: 4, 56: 43, 57: 43, 58: 28, 59: 12, 60: 32, 61: 8, 62: 43, 63: 23, 64: 6, 65: 6, 66: 9, 68: 27, 69: 8, 70: 33, 71: 28, 72: 5, 73: 21, 74: 6, 75: 20, 76: 9, 77: 2, 78: 44, 79: 29}
    {0: 33, 2: 32, 3: 1, 4: 1, 5: 3, 7: 17, 8: 3, 9: 3, 10: 12, 11: 1, 12: 19, 13: 18, 14: 12, 15: 12, 17: 1, 20: 3, 24: 12, 25: 7, 26: 23, 27: 19, 28: 15, 29: 7, 30: 6, 31: 11, 32: 23, 33: 4, 34: 33, 35: 4, 36: 5, 38: 3, 39: 26, 40: 8, 41: 11, 42: 50, 43: 25, 44: 21, 45: 41, 46: 1, 47: 14, 48: 4, 49: 3, 50: 28, 51: 22, 52: 5, 55: 9, 56: 60, 57: 22, 58: 13, 59: 14, 60: 44, 61: 15, 62: 3, 63: 18, 64: 7, 65: 2, 66: 1, 67: 5, 68: 34, 69: 9, 70: 49, 71: 35, 72: 14, 73: 14, 74: 9, 75: 7, 76: 1, 77: 9, 78: 49, 79: 24}

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train",
        examples_per_class=examples_per_class,
        synthetic_probability=0,
        use_randaugment=False,  # Test with True
        seed=seed,
        image_size=(image_size, image_size)
    )

    # Calculate class weights based on the inverse of class frequencies. Assign weight to each sample in the dataset
    # based on the class distribution, so that each class has an equal contribution to the overall loss
    class_weights = 1.0 / train_dataset.class_counts
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

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=val_sampler, num_workers=4)

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

    filter_model.load_state_dict(best_filter_model)

    os.makedirs(model_dir, exist_ok=True)
    log_path = f"logs/train_filter_{seed}_{epoch + 1}x{iterations_per_epoch}.csv"
    model_path = f"{model_dir}/ClassificationFilterModel.pth"

    pd.DataFrame.from_records(records).to_csv(log_path)
    torch.save(filter_model, model_path)

    print(f"Model saved to {model_path} - Validation loss {best_validation_loss} - Validation accuracy "
          f"{corresponding_validation_accuracy} - Training results saved to {log_path}")


class ClassificationFilterModel(nn.Module):

    def __init__(self, num_classes: int):
        super(ClassificationFilterModel, self).__init__()

        self.image_processor = None
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Filter")

    parser.add_argument("--examples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    args = parser.parse_args()

    train_filter(examples_per_class=args.examples_per_class,
                 seed=args.seed,
                 weight_decay=args.weight_decay)

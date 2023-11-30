from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, DeiTModel
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

DATASETS = {
    "spurge": SpurgeDataset,
    "coco": COCODataset,
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset
}


def train_filter(seed: int = 0,
                 dataset: str = "coco",
                 iterations_per_epoch: int = 200,
                 num_epochs: int = 50,
                 batch_size: int = 32,
                 image_size: int = 256,
                 classifier_backbone: str = "resnet50"):
    """
    Trains a classifier on the train data and saves the version with the highest validation accuracy.
    This model can then be used to filter synthetic images later on.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train",
        synthetic_probability=0,
        synthetic_dir=None,
        seed=seed,
        image_size=(image_size, image_size))

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=4)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size))

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=val_sampler, num_workers=4)

    filter_model = ClassificationModel(
        train_dataset.num_classes,
        backbone=classifier_backbone
    ).cuda()

    optim = torch.optim.Adam(filter_model.parameters(), lr=0.0001)

    for epoch in trange(num_epochs, desc="Training Filter"):

        filter_model.train()

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
            if len(label.shape) > 1:
                label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():

                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)
        training_accuracy = training_accuracy.cpu().numpy()

        filter_model.eval()

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
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)
        validation_accuracy = validation_accuracy.cpu().numpy()

    print('training accuracy of filter', training_accuracy.mean())
    print('validation accuracy of filter', validation_accuracy.mean())

    return filter_model


class ClassificationModel(nn.Module):

    def __init__(self, num_classes: int, backbone: str = "resnet50"):

        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor = None

        if backbone == "resnet50":

            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)

        elif backbone == "deit":

            self.base_model = DeiTModel.from_pretrained(
                "facebook/deit-base-distilled-patch16-224")
            self.out = nn.Linear(768, num_classes)

    def forward(self, image):

        x = image

        if self.backbone == "resnet50":

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

        elif self.backbone == "deit":

            with torch.no_grad():

                x = self.base_model(x)[0][:, 0, :]

        return self.out(x)

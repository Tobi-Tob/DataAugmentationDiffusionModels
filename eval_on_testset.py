import os

import pandas as pd

from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.road_sign import RoadSignDataset
from semantic_aug.datasets.coco_extension import COCOExtension
from semantic_aug.datasets.focus import FOCUS
from train_classifier import ClassificationModel

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse


DATASETS = {
    "coco": COCODataset,
    "road_sign": RoadSignDataset,
    "coco_extension": COCOExtension,
    "focus": FOCUS,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Eval on Testset")

    parser.add_argument("--dataset", type=str, default="coco_extension", choices=DATASETS.keys())
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model", type=str, default="models/classifier_coco_extension_0_8_[0.7]_[15.0].pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epc", type=int, default=0)

    parser.add_argument("--logdir", type=str, default="logs")

    """
    MR:
    python eval_on_testset.py --dataset "coco_extension" --split "test_uncommon" --model "RESULTS/coco_extension_2epc/paper/models/classifier_coco_extension_0_2_[0.5]_[7.5].pth" --seed 0 --epc 2 --logdir "RESULTS/coco_extension_2epc/paper/uncommon_test/"
    
    python eval_on_testset.py --dataset "coco_extension" --split "test_uncommon" --model "models/classifier_coco_extension_0_8_[0.7]_[15.0].pth" --seed 0 --epc 8 --logdir "logs"
    
    """

    args = parser.parse_args()
    # Model:
    # Should be ClassificationModel as defined in train_classifier
    model_path = args.model
    # Dataset:
    dataset = args.dataset
    split = args.split
    epc = args.epc
    seed = args.seed
    image_size = 256
    # Log dir
    logdir = args.logdir

    # Build the test dataset
    test_dataset = DATASETS[dataset](split=split, seed=seed, image_size=(image_size, image_size))
    test_dataloader = DataLoader(test_dataset)

    print(f'Loading model...')
    print(model_path)
    model = ClassificationModel(test_dataset.num_classes, backbone="resnet50")
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    print(f'Evaluating on {split} dataset...')
    epoch_loss = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
    epoch_accuracy = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
    epoch_size = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
    for image, label in test_dataloader:
        image, label = image.cuda(), label.cuda()
        logits = model(image)
        prediction = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, label, reduction="none")
        accuracy = (prediction == label).float()
        with torch.no_grad():
            epoch_size.scatter_add_(0, label, torch.ones_like(loss))
            epoch_loss.scatter_add_(0, label, loss)
            epoch_accuracy.scatter_add_(0, label, accuracy)

    test_loss = epoch_loss / epoch_size.clamp(min=1)
    test_accuracy = epoch_accuracy / epoch_size.clamp(min=1)
    test_loss = test_loss.cpu().numpy()
    test_accuracy = test_accuracy.cpu().numpy()
    print(f'{split} accuracy: {test_accuracy.mean()}')

    testset_record = [dict(value=test_loss.mean(), metric=f"Mean Loss"),
                      dict(value=test_accuracy.mean(), metric=f"Mean Accuracy")]
    for i, name in enumerate(test_dataset.class_names):
        testset_record.append(dict(value=test_loss[i], metric=f"Loss {name.title()}"))
        testset_record.append(dict(value=test_accuracy[i], metric=f"Accuracy {name.title()}"))
    test_path = os.path.join(logdir, f"evaluation_on_{dataset}_{split}_{seed}_{epc}.csv")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    pd.DataFrame.from_records(testset_record).to_csv(test_path)
    print(f"Evaluation saved to: {test_path}")


from torch.utils.data import Dataset
from typing import Any, Tuple, List
from PIL import Image

import torch.nn as nn
import torch
import abc


class GenerativeAugmentation(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[List[Image.Image], int]: # MR: changed output from Tuple[] to List[Tuple[]]

        return NotImplemented
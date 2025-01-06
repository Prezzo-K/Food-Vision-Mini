"""
Contains the functionality to to take a path of standard image classification
dataset and convert it to a Dataset and then Dataloaders.
"""

import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pathlib
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any

def data_setup(data_dir: pathlib.Path,
               train_transform: torchvision.transforms,
               test_transform: torchvision.transforms = None,
               batch_size: int = 32,
               ) -> Tuple[torch.utils.data.DataLoader,
                          torch.utils.data.DataLoader,
                          List[str],
                          Dict[str, int]
                          ]:
  """Converts custom data to tochvision Dataset then to Dataloaders.

  Contains the code to take a standard image classification dataset and convert
  it to a torchvision image classification dataset then to a training dataloader
  and testing dataloader.

  Args:
      data_dir: The directory that contains the train data and test data.
      train_transform: The transform to be applied to the train data.
      test_transform: The transform to be applied to the test data. If None, then
          train_transform is used to transfrom the test data.
      batch_size: The batch size to be applied to dataloaders.

  Returns:
      A tuple of (train_dataloader, test_dataloader, class_names, classes_to_idx)
      example usage:
          train_dataloader, test_Dataloader, class_names, class_to_idx =
              data_setup(data_dir  = <path_to_data_dir>,
                         train_transform = <train_transform>,
                         test_transform = <test_transform>,
                         batch_size = <batch_size> = 32 # default is 32 if unspecified.
                         )
  """


  # create train and test dirs
  train_dir = Path(os.path.join(data_dir, "train"))
  test_dir = Path(os.path.join(data_dir, "test"))

  # create the train dataset and test dataset using inbuilt ImageFolder.
  train_data = ImageFolder(root = train_dir,
                           transform = train_transform,
                           target_transform = None
                           )
  if test_transform is not None:
    test_data = ImageFolder(root = test_dir,
                            transform = test_transform,
                            target_transform = None
                            )
  else:
    test_data = ImageFolder(root = test_dir,
                            transform = train_transform,
                            target_transform = None
                            )

  # Create the training dataloaders and testing dataloaders.
  NUM_WORKERS = os.cpu_count()


  train_dataloader = DataLoader(dataset = train_data,
                                batch_size = batch_size,
                                num_workers = NUM_WORKERS,
                                pin_memory = True,
                                shuffle = True
                                )
  test_dataloader = DataLoader(dataset = test_data,
                               batch_size = batch_size,
                               num_workers = NUM_WORKERS,
                               pin_memory = True,
                               shuffle = False
                               )

  # get class data and other attributes to return
  class_names = train_data.classes
  classes_to_idx = test_data.class_to_idx

  return train_dataloader, test_dataloader, class_names, classes_to_idx

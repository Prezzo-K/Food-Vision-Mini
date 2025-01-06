
"""This file contains helper functions and other functions that are used oftenly
in model creation, debuuging and deployment.
"""

import os
import pathlib
import subprocess

import torch

from pathlib import Path

def save_model(model: torch.nn.Module,
               model_name: str,
               model_dir_name: str = "models",
               project_dir_name: str  ="going_modular",
               ) -> None:
  """Saves a pytorch in to user specified model_dir under project_dir

  Contains the functionality to save a pytorch model with th  extension .pth into
  a specified user dir i.e linearModel.pth under models directory.

  Args:
    model: The pytorch model to be saved.
    model_name: The name of the model to be saved. Default is "models"
    model_dir: The name of directory in which the model will be saved.
    project_dir: The name of the project_dir for path concatenation. by default 
    it is set to "going_modular".
    example usage:
        save_model(model = <some_model>,
                   model_name = <some_model_name>,
                   model_dir = <some_model_dir_name>,
                   project_dir = <some_project_dir_name>
                   )

  Returns:
    None.
  
  Raises:
    Raises and AssertionError if model_name doesn't end with .pth extension.
  """

  # check if model_name ends with .pth extension
  assert model_name.endswith(".pth"), f"{model_name} should end with .pth"
  # Create the path dir directory
  save_path = Path(os.path.join(project_dir_name, model_dir_name))

  # check if path already exist to avoid override
  if os.path.exists(save_path):
    overwrite = input(f"{save_path} already exists. Overwrite? (y/n): ")
    if overwrite.lower() != ("y"):
      print(f"Model saving canceled.")
    else:
      new_path = Path(input(f"Please provide a new path: "))
      save_path = Path(os.path.join(project_dir_name, new_path))
      save_path.mkdir(parents = True, exist_ok = True)
      save_path = save_path / model_name
    
      # save model under new dir
      torch.save(obj = model.state_dict(), f = save_path)
      print(f"Model is saved to {save_path}")
  else:
    # save model
    save_path.mkdir(parents = True, exist_ok = True)
    save_path = save_path / model_name
    torch.save(obj = model.state_dict(), f = save_path)
    print(f"Model is saved to {save_path}")

def visualize_model_architecture(model: torch.nn.Module,
                                 input_size: torch.Tensor):
  """This function return a display of model architecture.

  Contains functionality to display a model architecture. Returns 
  torchinfo.model_statistics.ModelStatistics object. import neccessary modules
  at runtime.

  Args:
    model: The model to be displayed its architecture.
    input_size: A sample input_size i.e a single feature or a batch.
  
  Returns:
    torchinfo.model_statistics.ModelStatistics object
  """
  # install torchinfo at runtime
  subprocess.run(["pip", "install", "torchinfo"])
  # import torchinfo and its summary
  import torchinfo
  from torchinfo import summary

  # return the object summary
  return summary(model = model,
          input_size = input_size,
          col_width = 25,
          col_names = ["input_size", "output_size", "num_params", "trainable"],
          row_settings = ["var_names"])

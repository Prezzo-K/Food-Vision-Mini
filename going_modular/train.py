"""This is a python script that downloads standard image classification data from
["https://github.com/mrdbourke/pytorch-deep-learning/raw/6aa8b4c9f868c6fc6113badce
81b38cda71dfd30/data/pizza_steak_sushi_20_percent.zip"] and set ups the project dir
and all other meta data.

The scripts then create a TinyVGG model and trains and evaluates on the data.
"""

import os
import subprocess

import torch
import torchvision

from torchvision.transforms import v2
from pathlib import Path
from IPython.display import display

from going_modular.download_data import download_data
from going_modular.data_setup import data_setup
from going_modular.model_builder import TinyVGG
from going_modular.utils import visualize_model_architecture, save_model
from going_modular.engine import train_test_step


print("[INFo] setting up device agnostic code........\n")
device = "cuda" if torch.cuda.is_available() else "cpu"

# download the data and set up
github_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/6aa8b4c9f868c6fc6113badce81b38cda71dfd30/data/pizza_steak_sushi_20_percent.zip"
download_data(project_dir = "going_modular",
              root_dir = "data",
              github_url  = github_url
              )

print(f"\n[INFO] creating a torchvision dataset with dataloaders.....")
# specify data_dir to create test and train dir
project_dir = "going_modular"
data_dir = "data"
sub_data_dir = "pizza_steak_sushi"

data_dir = Path(os.path.join(project_dir, data_dir, sub_data_dir))

# create transforms for the train data and test data
train_transform = v2.Compose([v2.Resize(size = (64,64)),
                              v2.TrivialAugmentWide(num_magnitude_bins = 32),
                              v2.ToImage(),
                              v2.ToDtype(dtype = torch.float32,
                                         scale = True)
                              ])
test_transform = v2.Compose([v2.Resize(size = (64,64)),
                             v2.ToImage(),
                             v2.ToDtype(dtype = torch.float32,
                                        scale = True)
                             ])

# Set up the dataset and dataloaders
BATCH_SIZE = 16

train_dataloader, test_dataloader, class_names, classes_to_idx = data_setup(
    data_dir = data_dir,
    train_transform = train_transform,
    test_transform = test_transform,
    batch_size = BATCH_SIZE
)

# Print process
print(f"\n[INFO] created a torchvision dataset and train dataloaders and test dataloaders")
print(f"[INFO] class names are: {class_names}\n")

print("[INFO] creating TinyVGG model from scratch............")
print("[INFO] downloading neccessary packages too ..............")

# set up hyperparameters for the model
INPUT_FEATURES = 3 # for RGB images
HIDDEN_UNITS = 12
OUTPUT_FEATURES = len(class_names)

# create the model
model_0 = TinyVGG(input_features = INPUT_FEATURES, hidden_units = HIDDEN_UNITS, output_features = OUTPUT_FEATURES).to(device)
print(f"[INFO] created a TinyVGG model. Visualizing..............\n")

# visualize the model architectures
sample_pass_input_shape = torch.randn(size = (32,3,64,64)).shape # batch size but can be a single feature.
summary = visualize_model_architecture(model = model_0,
                                       input_size = sample_pass_input_shape
                                       )
# visualize using in-built disply
display(summary)

# setting loss function, optimizer and accuracy function
print(f"[INFO] setting a loss function, optimizer and accuracy fucntion\n")

subprocess.run(["pip", "install", "torchmetrics"])
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

loss_fn = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1e-02
optimizer = torch.optim.Adam(params = model_0.parameters(), lr = LEARNING_RATE)
accuracy_fn = MulticlassAccuracy(num_classes = len(class_names))

# train the model
EPOCHS = 5

print(f"[INFO] starting to train the model...........\n")
model_results = train_test_step(model = model_0,
                                train_dataloader = train_dataloader,
                                test_dataloader = test_dataloader,
                                loss_fn = loss_fn,
                                optimizer = optimizer,
                                accuracy_fn = accuracy_fn,
                                device = device,
                                epochs = EPOCHS
                                )

# save the model
print(f"\n[INFO] training done.... Saving the model............\n")
save_model(model = model_0,
           model_name = "TinyVGG.pth"
           )

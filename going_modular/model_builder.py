import torch
from torch import nn

class TinyVGG(nn.Module):
  """A baseline class to replicate TinyVGG model.

  Contains all replica info, layers and other meta data to replicate TinyVGG
  architecture.
  """
  def __init__(self, input_features : int, hidden_units: int, output_features: int):
    """Create instance of the model with necessary arguments.

    Defines the layout of the architecture i.e blocks, Conv2d layers, activation functions,
    and other layers i.e MaxPool.

    Args:
        input_features: The input channels of the model. In this case 3 for RGB images.
        hidden_units: The number of neurons of each layer in the model.
        output_features: The number of output features of the model. 3 in this case for
            ["pizza", "steak", "sushi"]

    """
    # Inherit from the base class
    super().__init__()

    # Define the architecture of block 1 of the model.
    self.block_1 = nn.Sequential(nn.Conv2d(in_channels = input_features, out_channels = hidden_units,
                                           kernel_size = (3,3),
                                           stride = 1,
                                           padding = 1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                                           kernel_size = (3,3),
                                           stride = 1,
                                           padding = 1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size = (2,2),
                                              stride = 2),
                                 )

    # Define the architecture of block 2 of the model.
    self.block_2 = nn.Sequential(nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                                           kernel_size = (3,3),
                                           stride = 1,
                                           padding = 1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                                          kernel_size = (3,3),
                                          stride = 1,
                                          padding = 1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size = (2,2),
                                              stride = 2),
                                 )

    # Define the architecture of the classifier head
    self.classifier = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features = hidden_units * 16 * 16, out_features = output_features)
                                    )

  # Override the forward method
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """A fucntion that does forward computation at one go.

    Computes the logits by doing forward propagation of data through the model
    using operator fusion to improve efficency.

    Args:
        x: The torch.Tensor input that is forward propagated through the model.

    Returns:
        logit prediction of size [bacth_size, len(class_names)] for every pass.
        Example usage:
            model_0 = TinyVGG(input_features = 3, hidden_units = 12,
                              output_features = len(class_names)
                              )
            logit = model_0(torch.randn(size = (32, 3, 64, 64))
                            )
    """
    """
    x = self.block_1(x)
    print(f"After block_1: {x.shape}")
    x = self.block_2(x)
    print(f"After block_2: {x.shape}")
    x = self.classifier(x)
    print(f"After classifier head: {x.shape}")
    return x
    """
    return self.classifier(self.block_2(self.block_1(x)))

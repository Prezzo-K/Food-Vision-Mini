"""Contains the functionality to train and test a pytorch model in one hit.
"""
import torch
import torchmetrics

from typing import List, Dict, Tuple, Callable
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.classification.MulticlassAccuracy,
               device: torch.device,
               ) -> Tuple[float, float]:
  """A fucntion to train a pytorch model over the train data.

  Contains functionality to train a pytorch model and to return the loss and accuracy
  after each epoch.

  Args:
      model: The pytorch model to be trained.
      train_dataloader: The training dataloader to iterated over.
      loss_fn: The loss fucntion that determines how bad our model is.
      optimizer: The optimizer to do optimization and weight updates.
      accuracy_fn: A metric that calculates the accuracy of our model.
      device: A runtime environment. cuda or cpu.

  Returns:
      A tuple of (train_loss, train_acc) that says how model is performing.
      Example usage:
          train_loss, train_acc = train_step(model = <some_model>,
                                             train_dataloader = <some_dataloader>,
                                             optimizer = <some_optimizer>,
                                             accuracy_fn = <some_accuracy_fn>,
                                             device = <some_device>
                                             )
  """

  # define a train loss and a test loss
  train_loss, train_acc = 0, 0

  # the model to train mode
  model.train()
  # loop over the train_dataloader
  for batch, (X, y) in enumerate(train_dataloader):
    # send data to device
    X, y = X.to(device), y.to(device) # can be optimized in one hit instead of every batch!
    # do a forward pass and calculate the loss
    logits = model(X)
    loss = loss_fn(logits, y)
    # calculate the preds by doing softmax at dim 1 then argmax to get the idicies
    train_preds = torch.softmax(logits, dim = 1).argmax(dim = 1)

    # accumulate the train loss and accuracy over the batches
    train_loss += loss.item()
    train_acc += accuracy_fn(train_preds, y).item()

    # Set the optimizer to zero grad
    optimizer.zero_grad()
    # Do backprogation
    loss.backward()
    # Do gradient descent
    optimizer.step()

    # print out what is happenning
    if batch % 3 == 0:
      print(f"Looked at {batch * len(X)} training image samples...")

  # get the average loss and accuracy over the entire epoch
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: torchmetrics.classification.MulticlassAccuracy,
              device:torch.device
              ) -> Tuple[float, float]:
  """A fucntion to test a pytorch model over the test data.

  Contains functionality to train a pytorch model and to return the loss and accuracy
  after each epoch.

  Args:
      model: The pytorch model to be trained.
      test_dataloader: The testing dataloader to iterated over.
      loss_fn: The loss fucntion that determines how bad our model is.
      accuracy_fn: A metric that calculates the accuracy of our model.
      device: A runtime environment. cuda or cpu.

  Returns:
      A tuple of (test_loss, test_acc) that says how model is performing.
      Example usage:
          test_loss, test_acc = test_step(model = <some_model>,
                                             test_dataloader = <some_dataloader>,
                                             loss_fn = <soem_loss_fn>,
                                             accuracy_fn = <some_accuracy_fn>,
                                             device = <some_device>
                                             )
  """

  # define test_loss and test_acc
  test_loss, test_acc = 0 ,0

  # set the model to eval mode
  model.eval()
  with torch.inference_mode():
    # loop over the test dataloader
    for X, y in test_dataloader:
      # send data to device
      X, y = X.to(device), y.to(device) # can be optimized in one hit instead of every batch!
      # do a forward pass and calculate the loss
      test_logits = model(X)
      test_loss += loss_fn(test_logits, y).item()

      # get the preds nd accumaulate the accuracy
      test_preds = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
      test_acc += accuracy_fn(test_preds, y).item()

  # get the average over all batches
  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)

  return test_loss, test_acc

def train_test_step(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    test_dataloader: torch.utils.data.DataLoader,
                    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    accuracy_fn: torchmetrics.classification.MulticlassAccuracy,
                    device: torch.device,
                    epochs: int = 5
                    ) -> Dict[str, List[float]]:
  """A fucntion to test and test pytorch model in one hit. Coombines train_step
  fuction and test_step function.

  Contains functionality to train a pytorch model  and test it. Logs all the
  neccessary mertics and returns them in a dictionary.

  Args:
      model: The pytorch model to be trained.
      train_dataloader: The training dataloader to be iterated over.
      test_dataloader: The testing dataloader to iterated over.
      loss_fn: The loss fucntion that determines how bad our model is.
      optimizer: The optimizer to be used for gradient descent.
      accuracy_fn: A metric that calculates the accuracy of our model.
      device: A runtime environment. cuda or cpu.
      epochs: the number of epochs to train

  Returns:
      A dictionary of {"train_loss": [], "train_acc": [],
                       "test_loss": [], "test_acc": []
                       }
      This contains all the metrics that were logged during training and
          evaluating.
      Example usage:
          r=model_results = train_test_step(model = <some_model>,
                                             train_dataloader = <some_dataloader>,
                                             test_dataloader = <some_dataloader>,
                                             loss_fn = <some_loss_fn>,
                                             optimizer = <some_optimizer>,
                                             accuracy_fn = <some_accuracy_fn>,
                                             device = <some_device>,
                                             epochs = <epochs>
                                             )
  """

  # setup resuslts dictionary
  model_results = {"train_loss" : [],
                   "train_acc" : [],
                   "test_loss" : [],
                   "test_acc" : []
                   }

  # loop over the epochs
  for epoch in tqdm(range(epochs), desc = "Model Training In Progress >>>>>>>>>>>>>>>>"):
    # print which epoch
    print(f"\nEpoch: {epoch + 1}\n_____________________")

    # do pass in the train step
    train_loss, train_acc = train_step(model = model,
                                     train_dataloader = train_dataloader,
                                     loss_fn = loss_fn,
                                     optimizer = optimizer,
                                     accuracy_fn = accuracy_fn,
                                     device = device
                                      )

    # do a pass in test step
    test_loss, test_acc = test_step(model = model,
                                test_dataloader = test_dataloader,
                                loss_fn = loss_fn,
                                accuracy_fn = accuracy_fn,
                                device = device
                                  )

    # print out the losses and accuracies in each epoch
    print(f"\nTrain_loss: {train_loss:.5f} | Train_acc: {train_acc:2f} | Test_loss: {test_loss:.5f} | Test_acc: {test_acc:.2f}")

    # log them in dictionary
    model_results["train_loss"].append(train_loss)
    model_results["train_acc"].append(train_acc)
    model_results["test_loss"].append(test_loss)
    model_results["test_acc"].append(test_acc)

  # return the dictionary
  return model_results

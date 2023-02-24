import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import logging


def create_validation_metrics(loader, net):
    """Create validation metrics in training loop.

    Args:
        loader (DatasetLoader): Validation dataset loader.
        net (nn.Module): Model

    Returns:
        Dict: Dictionary of confusion matrix plot and accuracy.
    """
    y_pred = []  # save predction
    y_true = []  # save ground truth

    # iterate over data
    with torch.no_grad():
        for inputs, labels in loader:
            output = net(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth
    # constant for classes
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    accuracy = np.sum(np.array(y_pred) == np.array(y_true)) / len(y_pred)
    df_cm = pd.DataFrame(
        cf_matrix, index=[i for i in classes], columns=[i for i in classes]
    )
    plt.figure(figsize=(12, 7))
    return {
        "confusion_plot": sn.heatmap(df_cm, annot=True).get_figure(),
        "accuracy": accuracy,
    }


class CustomMnistDataset(TensorDataset):
    """Dataset creator for MNIST kaggle dataset"""

    def __init__(self, img_dir, type, transform=None):
        if type == "train":
            self.labels = True
            filename = "train.csv"
        elif type == "validation":
            self.labels = True
            filename = "validation.csv"
        elif type == "test":
            self.labels = False
            filename = "test.csv"
        else:
            raise ValueError(
                "Keyword arguement for type should be in {'train', 'validation', 'test'}"
            )
        self.transform = transform
        fp = Path(img_dir) / filename  # type: ignore
        ds = pd.read_csv(fp, dtype=np.float32)
        X_numpy = ds.loc[:, ds.columns != "label"].values  # type: ignore
        X_numpy_normalized = X_numpy / 255.0
        X = torch.from_numpy(X_numpy_normalized.reshape(-1, 1, 28, 28))

        if self.labels:
            y_numpy = ds.label.values
            y_train = torch.from_numpy(y_numpy).type(torch.LongTensor)  # type: ignore
            super(CustomMnistDataset, self).__init__(X, y_train)
        else:
            super(CustomMnistDataset, self).__init__(X)

    def __getitem__(self, index):
        data = super(CustomMnistDataset, self).__getitem__(index)
        if self.transform:
            data = list(data)
            data[0] = self.transform(data[0])  # type: ignore
            return tuple(data)
        return data


def imshow(inp, title=None, cmap=None):
    """Show image helper"""
    inp = inp.permute((1, 2, 0))
    plt.imshow(X=inp[:, :, 0], cmap=cmap)  # type: ignore
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def train_covnet(
    net, dataloader, epochs, optimizer_wrapper, criterion, writer, val_dataloader=None
):
    """Training convolutional network loop

    Args:
        net (nn.Module): The model to train
        dataloader (DatasetLoader): Torch Dataset loader for training.
        epochs (int): Number of epochs
        optimizer_wrapper (function): Function taking a network parameters and returns an optimizer.
        criterion (nn.Loss): Criterion for training.
        writer (SummaryWriter): Tensorboard summary writter.
        val_dataloader (DatasetLoader, optional): Validation set dataset loader. If None
         skip logging of validation metrics Defaults to None.

    Returns:
        Dict: Dictionary of trained network, optimizer, criterion, and writer.
    """
    optimizer = optimizer_wrapper(net.parameters())
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        accuracy = 0.0
        N = len(dataloader)
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # save prediction
            accuracy += torch.sum(preds == labels.data)
            if i % 500 == 499:  # print every 500 mini-batches
                steps = epoch * N + i  # calculate steps
                accuracy_out = accuracy / (500 * dataloader.batch_size)
                logging.info(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}, accuracy: {accuracy_out:.3f}"
                )
                writer.add_scalar("Training loss by steps", running_loss / 500, steps)
                writer.add_scalar(
                    "Training accuracy by steps",
                    accuracy_out,
                    steps,
                )
                accuracy = 0.0
                running_loss = 0.0
        if val_dataloader:
            net.eval()
            validation_metrics = create_validation_metrics(val_dataloader, net)
            writer.add_figure(
                "Validation Confusion matrix",
                validation_metrics.get("confusion_plot"),
                epoch + 1,
            )
            writer.add_scalar(
                "Validation accuracy by epoch",
                validation_metrics.get("accuracy"),
                epoch + 1,
            )
            net.train()
    return {
        "model": net,
        "optimizer": optimizer,
        "criterion": criterion,
        "writer": writer,
    }

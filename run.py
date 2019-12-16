"""PyTorch Implementation of LeNet-5, Recreating the network in http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
Uses ReLU instead of tanh activation
Uses Max pooling instead of average
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import mlflow
import mlflow.pytorch

from model import LeNet5
from data import DigitDataset, ToTensor, Normalize, ZeroPad


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(model, images):
    prediction = model(images)
    _, preds_tensor = torch.max(prediction, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, F.softmax(prediction)[0][preds]


def plot_classes_preds(model, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(model, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(1):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            preds,
            probs * 100.0,
            labels),
            color=("green" if preds==labels.cpu().numpy() else "red"))
    return fig


def log_scalar(writer, name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


if __name__ == '__main__':
    validation_split = 0.1
    shuffle_dataset = True
    random_seed = 42
    batch_size = 1
    initial_learning_rate = 0.05
    sgd_momentum = 0.9
    running_loss = 0.0
    epochs = 10
    current_epoch = 0
    epoch_start_time = None
    summary_output_path = 'runs/digit_recognizer_experiment_8'
    writer = SummaryWriter(summary_output_path)

    data_transforms = transforms.Compose([ZeroPad(pad_size=2),
                                          # Normalize(0.1310, 0.308),
                                          ToTensor()])
    data = DigitDataset('data/train.csv', transform=data_transforms, train=True)
    # Creating data indices for training and validation splits:
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler)
    data_test = DigitDataset('data/test.csv', transform=data_transforms, train=False)

    # get some random training images
    dataiter = iter(train_loader)
    sample = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(sample['image'])

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_digit_images', img_grid)

    model = LeNet5()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=sgd_momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[2, 5, 8, 12],
                                         gamma=0.1)
    # loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
    loss_fn = torch.nn.MSELoss()

    if torch.cuda.is_available():
        print("Using GPU")
        model.cuda()

    writer.add_graph(model, sample['image'])
    running_loss = 0.0
    start_time = time.time()
    start_epoch = 0
    with mlflow.start_run():
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('initial_learning_rate', initial_learning_rate)
        mlflow.log_param('sgd_momentum', sgd_momentum)
        mlflow.log_param('validation_split', validation_split)
        # TRAIN
        for current_epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            model.train(True)
            for batch_index, sample in enumerate(train_loader):
                image = Variable(sample['image'])
                optimizer.zero_grad()
                # TODO: Detect loss type and do the right transformation on label
                label = Variable(sample['label'])
                label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=10).float().cuda()
                y_pred = model(image)
                loss = loss_fn(y_pred, label_one_hot)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_index % 1000 == 999:
                    step = current_epoch * len(train_loader) + batch_index
                    log_scalar(writer, 'training loss', running_loss / 1000, step)
                    writer.add_figure('predictions vs. actuals',
                                      plot_classes_preds(model, image, label),
                                      global_step=step)
                    model.log_weights(step, writer)
                    running_loss = 0.0

            scheduler.step()

            # TEST
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():

                for sample in validation_loader:
                    image = Variable(sample['image'])
                    label = Variable(sample['label'])
                    label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=10).float().cuda()
                    output = model(image)
                    test_loss += loss_fn(output, label_one_hot).item()
                    prediction = output.data.max(1)[1]
                    correct += prediction.eq(label.data).cpu().sum().item()

                test_loss /= len(validation_loader)
                test_accuracy = 100 * correct/len(validation_loader)
                step = (current_epoch + 1) * len(train_loader)
                log_scalar(writer, 'test loss', test_loss, step)
                log_scalar(writer, 'test accuracy', test_accuracy, step)
        # mlflow.log_artifacts(summary_output_path, artifact_path='events')
        mlflow.pytorch.log_model(model, "models")

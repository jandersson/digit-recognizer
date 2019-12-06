"""PyTorch Implementation of LeNet-5, Recreating the network in http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
Uses ReLU instead of tanh activation
Uses Max pooling instead of average
"""

import argparse
from datetime import datetime
import logging
import time

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io, transform


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)




def update_learning_rate(optimizer, current_epoch, override=None):
    """Deprecated: Return optimizer with learning rate schedule from paper"""
    # Learning Rate schedule: 0.0005 for first 2 iterations, 0.0002 for next 3, 0.0001 next 3, 0.00005 next 4,
    # 0.00001 thereafter
    if current_epoch < 2:
        new_lr = 5e-4
    elif current_epoch < 5:
        new_lr = 2e-4
    elif current_epoch < 8:
        new_lr = 1e-4
    elif current_epoch < 12:
        new_lr = 5e-5
    else:
        new_lr = 1e-5
    if override:
        new_lr = override
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class DigitDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.digits_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.digits_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()
        label = self.digits_frame.iloc[idx, 0]
        image = self.digits_frame.iloc[idx, 1:].values
        image = image.reshape(28, 28)
        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class LeNet5(torch.nn.Module):
    """LeNet-5 CNN Architecture"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.c3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=120,
                            stride=1,
                            kernel_size=5,
                            padding=0),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        c3_flat = c3_out.view(c3_out.size(0), -1)
        return self.classifier(c3_flat)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image_dtype = torch.FloatTensor
        label_dtype = torch.LongTensor
        if torch.cuda.is_available():
            image_dtype = torch.cuda.FloatTensor
            label_dtype = torch.cuda.LongTensor
        image, label = sample['image'], sample['label']
        # add color channel so image is C x W x H
        image = image[np.newaxis, :, :]
        label = np.array([label])
        image = torch.from_numpy(image).type(image_dtype)
        label = torch.from_numpy(label).type(image_dtype)
        return {'image': image,
                'label': label}


class ZeroPad(object):
    def __init__(self, pad_size):
        self.pad_size = [(pad_size, pad_size), (pad_size, pad_size)]

    def __call__(self, sample):
        sample['image'] = np.pad(sample['image'], self.pad_size, mode='constant')
        return sample


class Normalize(object):
    """Make the mean input 0 and variance roughly 1 to accelerate learning"""
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def __call__(self, sample):
        original_shape = sample['image'].shape
        image = sample['image'].ravel()
        image -= self.mean
        image /= self.stdev
        image.shape = original_shape
        sample['image'] = image
        return sample






    # def test(self):
    #     """Tests model using test set"""
    #     self.model.train(False)
    #     correct = 0
    #     for sample in self.test_data:
    #         image = Variable(sample['image'])
    #         label = Variable(sample['label'])
    #         y_pred = self.model(image)
    #         correct += 1 if torch.equal(torch.max(y_pred.data, 1)[1], torch.max(label.data, 1)[1]) else 0
    #     test_accuracy = correct/len(self.test_data)
    #     self.vis.update_test_accuracy_plot(self.current_epoch + 1, test_accuracy)
    #     self.vis.write_log(f"Epoch: {self.current_epoch + 1}\tRunning Loss: {self.running_loss:.2f}\tEpoch time: {(time.time() - self.epoch_start_time):.2f} sec")
    #     self.vis.write_log(f"Test Accuracy: {test_accuracy:.2%}")
    #     self.vis.write_log(f"Elapsed time: {(time.time() - self.start_time):.2f} sec")


if __name__ == '__main__':
    import pandas as pd
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42
    batch_size = 1
    running_loss = 0.0
    epochs = 20
    current_epoch = 0
    epoch_start_time = None

    data_transforms = transforms.Compose([ZeroPad(pad_size=2),
                                          ToTensor()])
    data = DigitDataset('data/train.csv', transform=data_transforms)
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
    data_test = DigitDataset('data/test.csv')
    model = LeNet5()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[2, 5, 8, 12],
                                         gamma=0.1)
    # loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
    loss_fn = torch.nn.MSELoss()

    if torch.cuda.is_available():
        print("Using GPU")
        model.cuda()
    running_loss = 0.0
    start_time = time.time()
    start_epoch = 0
    # TRAIN
    for current_epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train(True)
        for batch_index, sample in train_loader:
            image = Variable(sample['image'])
            # TODO: Detect loss type and do the right transformation on label
            label = sample['label']
            label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=10).float()
            y_pred = model(image)
            loss = loss_fn(y_pred, label)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        running_loss += epoch_loss
        print(running_loss)

        # TEST
        model.train(False)
        correct = 0
        for sample in validation_loader:
            image = Variable(sample['image'])
            label = Variable(sample['label'])
            y_pred = model(image)
            correct += 1 if torch.equal(torch.max(y_pred.data, 1)[1], torch.max(label.data, 1)[1]) else 0
        test_accuracy = correct/len(data_test)
        print(f'accuracy: {test_accuracy}')
    # self.vis.update_test_accuracy_plot(current_epoch + 1, test_accuracy)
    # self.vis.write_log(f"Epoch: {self.current_epoch + 1}\tRunning Loss: {self.running_loss:.2f}\tEpoch time: {(time.time() - self.epoch_start_time):.2f} sec")
    # self.vis.write_log(f"Test Accuracy: {test_accuracy:.2%}")
    # self.vis.write_log(f"Elapsed time: {(time.time() - self.start_time):.2f} sec")


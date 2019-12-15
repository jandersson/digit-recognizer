from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class DigitDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        """Type can be train or test. Test has no label"""
        self.train = train
        self.digits_frame = pd.read_csv(csv_file, dtype=np.float64)
        self.transform = transform
        if train:
            self.mean = self.digits_frame.drop('label', axis=1).values.mean() / 255
            self.std = self.digits_frame.drop('label', axis=1).values.std(ddof=1) / 255
        else:
            self.mean = self.digits_frame.values.mean() / 255
            self.std = self.digits_frame.values.std(ddof=1) / 255

    def __len__(self):
        return len(self.digits_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()
        if self.train:
            label = self.digits_frame.iloc[idx, 0]
            image = self.digits_frame.iloc[idx, 1:].values
        else:
            label = None
            image = self.digits_frame.iloc[idx, :].values
        image = image.reshape(28, 28)
        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


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

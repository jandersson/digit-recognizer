import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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
        classifier_out = self.classifier(c3_flat)
        return F.softmax(classifier_out, dim=1)

    def log_weights(self, step: int, writer: SummaryWriter):
        writer.add_histogram('weights/c1/weight', self.c1[0].weight.data, step)
        writer.add_histogram('weights/c1/bias', self.c1[0].bias.data, step)
        writer.add_histogram('weights/c2/weight', self.c2[0].weight.data, step)
        writer.add_histogram('weights/c2/bias', self.c2[0].bias.data, step)
        writer.add_histogram('weights/c3/weight', self.c3[0].weight.data, step)
        writer.add_histogram('weights/c3/bias', self.c3[0].bias.data, step)
        writer.add_histogram('weights/classifier/weight', self.classifier[0].weight.data, step)
        writer.add_histogram('weights/classifier/bias', self.classifier[0].bias.data, step)

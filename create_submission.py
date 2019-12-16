from data import DigitDataset, ZeroPad, ToTensor
import mlflow.pytorch
from torchvision import transforms
import torch
import csv

data_transforms = transforms.Compose([ZeroPad(pad_size=2),
                                      # Normalize(0.1310, 0.308),
                                      ToTensor()])
test_data = DigitDataset('data/test.csv', train=False, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
run_id = '52a7abcb06e74c849f3a49db15149d1b'
model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")

model.eval()
with open('submission.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["ImageId", "Label"])
    with torch.no_grad():
        for batch_index, sample in enumerate(test_loader):
            output = model(sample['image'])
            prediction = output.data.max(1)[1].cpu().numpy()[0]
            writer.writerow([batch_index + 1, prediction])


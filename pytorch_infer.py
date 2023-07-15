import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set paths and parameters
data_path = "PetImages"  # Update with your actual data path
model_path = "cat_dog_classifier.pth"
device = "cpu"  # Change to "GPU" for NVIDIA GPU inference


class CatDogClassifier(pl.LightningModule):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)



# Load the model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = ImageFolder(data_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# do one inference to warmup
images, _ = next(iter(data_loader))
images = images.to(device)
outputs = model(images)


# Perform inference and measure throughput and FPS
total_time = 0
total_images = 0

with tqdm(total=len(data_loader), desc="Inference Progress") as pbar:
    for images, _ in data_loader:
        images = images.to(device)
        total_images += images.shape[0]

        # Start inference timer
        start_time = time.time()

        # Perform inference
        with torch.no_grad():
            outputs = model(images)

        # End inference timer
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        # Update progress bar
        pbar.update(1)

average_inference_time = total_time / total_images
fps = 1 / average_inference_time
throughput = total_images / total_time

# Print the results
print(f"Torch CPU Inference Time: {average_inference_time:.4f} seconds")
print(f"Torch CPU FPS: {fps:.2f}")
print(f"Torch CPUThroughput: {throughput:.2f} images/second")
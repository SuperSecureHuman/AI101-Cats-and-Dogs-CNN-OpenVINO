import time
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from openvino.inference_engine import IECore
from tqdm import tqdm

# Set paths and parameters
data_path = "./PetImages"  # Update with your actual data path
model_xml = "cat_dog_classifier.xml"
model_bin = "cat_dog_classifier.bin"
device = "CPU"  # Change to "GPU" for NVIDIA GPU inference

# Initialize the OpenVINO Inference Engine
ie = IECore()

# Load the OpenVINO IR model
net = ie.read_network(model=model_xml, weights=model_bin)

# Load the model to the specified device
exec_net = ie.load_network(network=net, device_name=device)

# Define input and output names
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and preprocess the images using PyTorch dataset loader
dataset = ImageFolder(data_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# do one inference to warmup
images, _ = next(iter(data_loader))
images = images.numpy()
outputs = exec_net.infer(inputs={input_blob: images})


# Start inference timer
start_time = time.time()

# Perform inference and measure throughput and FPS
total_time = 0
total_images = 0

with tqdm(total=len(data_loader), desc="Inference Progress") as pbar:
    for images, _ in data_loader:
        images = images.numpy()
        total_images += images.shape[0]

        # Start inference timer
        start_time = time.time()

        # Perform inference
        outputs = exec_net.infer(inputs={input_blob: images})

        # End inference timer
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        
        pbar.update(1)

average_inference_time = total_time / total_images
fps = 1 / average_inference_time
throughput = total_images / total_time

# Print the results
print(f"OpenVino Inference Time: {average_inference_time:.4f} seconds")
print(f"OpenVino Throughput: {throughput:.2f} images/second")
print(f"OpenVino FPS: {fps:.2f}")

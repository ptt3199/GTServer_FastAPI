from fastapi import APIRouter
from schema.dl_model import ModelInputSchema, ModelOutputSchema
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import asyncio
import psutil
import subprocess


router = APIRouter(prefix="", tags=['dl_model'])


def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_memory = [int(x) for x in output.strip().split('\n')]
        gpu_memory_usage_gb = [round(x / 1024, 2) for x in gpu_memory]
        return gpu_memory_usage_gb
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ModelClassification(nn.Module):
    def __init__(self, model_name=None, device="cpu"):
        super().__init__()
        self.device = device
        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        if model_name == 'resnet_18':
            model = models.resnet18(pretrained=False)
            num_classes = 10  # MNIST has 10 classes (digits 0-9)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            self.model = model
        elif model_name == 'resnet_50':
            model = models.resnet50(pretrained=False)
            num_classes = 10  # MNIST has 10 classes (digits 0-9)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            self.model = model
        else:
            raise "This model is not implemented!"
        self.model.to(torch.device(self.device))

    def forward(self, x):
        x = self.model(x)
        return x

    async def aync_predict_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(torch.device(self.device))
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        return predicted_class

    def predict_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(torch.device(self.device))
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        return predicted_class

# If we define the model as a global param, we might not leverate the gpu mem but we do not worry about the Out of Mem issue
#model = ModelClassification("resnet_18", "cuda")


@router.post("/inference")
async def inference(request_body: ModelInputSchema):
    # Get CPU and memory usage
    model = ModelClassification(request_body.model_name, request_body.device)
    print(f"Request ID: {request_body.request_id}, Start time: {datetime.now()},\
     CPU: {psutil.cpu_percent()} (%), GPU: {get_gpu_memory_usage()} (Gb), Memory (RAM) : {psutil.virtual_memory().percent} (%)")
    await asyncio.sleep(5)
    #output = model.predict_image(request_body.img_dir)
    output = await model.aync_predict_image(request_body.img_dir)
    print(output, type(output))
    print(f"Request ID: {request_body.request_id}, End time: {datetime.now()},\
         CPU: {psutil.cpu_percent()} (%), GPU: {get_gpu_memory_usage()} (Gb), Memory (RAM) : {psutil.virtual_memory().percent} (%)")
    return ModelOutputSchema(output=output)
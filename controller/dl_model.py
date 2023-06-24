from fastapi import APIRouter
from schema.dl_model import ModelInputSchema, ModelOutputSchema
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import asyncio

router = APIRouter(prefix="", tags=['dl_model'])


class ModelClassification(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        # Define the image transformation
        self.transform = transforms.Compose([
            #transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        if model_name == 'resnet_18':
            model = models.resnet18(pretrained=False)
            num_classes = 10  # MNIST has 10 classes (digits 0-9)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Load the pre-trained weights for the model
            #state_dict = torch.load("mnist_model_weights.pth")
            #model.load_state_dict(state_dict)
            self.model = model
        elif model_name == 'resnet_50':
            model = models.resnet50(pretrained=False)
            num_classes = 10  # MNIST has 10 classes (digits 0-9)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Load the pre-trained weights for the model
            #state_dict = torch.load("mnist_resnet50_model_weights.pth")
            #model.load_state_dict(state_dict)
            self.model = model
        else:
            raise "This model is not implemented!"

    def forward(self, x):
        x = self.model(x)
        return x

    async def predict_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image).unsqueeze(0)
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        return predicted_class


@router.post("/inference")
async def inference(request_body: ModelInputSchema):
    print(f"Request ID: {request_body.request_id}, Start time: {datetime.now()}")
    await asyncio.sleep(3)
    model = ModelClassification(request_body.model_name)
    output = await model.predict_image(request_body.img_dir)
    print(output, type(output))
    print(f"Request ID: {request_body.request_id}, End time: {datetime.now()}")
    return ModelOutputSchema(output=output)
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
from ultralytics import YOLO

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
    def __init__(self, model_name=None):
        super().__init__()
        if model_name == '2e':
            self.model = YOLO("./models/2e.pt")
        elif model_name == '4e':
            self.model = YOLO("./models/4e.pt")
        elif model_name == '6e':
            self.model = YOLO("./models/6e.pt") 
        elif model_name == '8e':
            self.model = YOLO("./models/8e.pt")
        elif model_name == '10e':
            self.model = YOLO("./models/10e.pt")
        elif model_name == '12e':
            self.model = YOLO("./models/12e.pt")
        elif model_name == '14e':
            self.model = YOLO("./models/14e.pt")
        elif model_name == '16e':
            self.model = YOLO("./models/16e.pt")
        elif model_name == '18e':
            self.model = YOLO("./models/18e.pt")
        elif model_name == '20e':
            self.model = YOLO("./models/20e.pt")
        else:
            raise "This model is not implemented!"

    def forward(self, x):
        x = self.model(x)
        return x

    async def aync_predict_image(self, image_path):
        results = self.model.predict(source=image_path)
        return results

    # def predict_image(self, image_path):
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')
    #     image = self.transform(image).unsqueeze(0)
    #     image = image.to(torch.device(self.device))
    #     # Perform inference
    #     with torch.no_grad():
    #         outputs = self.model(image)
    #         results = outputs.predict(source="./samples/hani2.jpg")
    #     return results


@router.post("/inference")
async def inference(request_body: ModelInputSchema):
    # Get CPU and memory usage
    model = ModelClassification(request_body.model_name)
    
    start_time = datetime.now()
    print(f"Request ID: {request_body.request_id}, Start time: {start_time},\
        CPU: {psutil.cpu_percent()} (%), GPU: {get_gpu_memory_usage()} (Gb), Memory (RAM) : {psutil.virtual_memory().percent} (%)")
    # await asyncio.sleep(5)
    
    output = await model.aync_predict_image(request_body.img_dir)
    # print(output)
    
    end_time = datetime.now()
    print(f"Request ID: {request_body.request_id}, End time: {end_time},\
        CPU: {psutil.cpu_percent()} (%), GPU: {get_gpu_memory_usage()} (Gb), Memory (RAM) : {psutil.virtual_memory().percent} (%)")
    
    processing_time = end_time - start_time

    return ModelOutputSchema(output=str(processing_time))
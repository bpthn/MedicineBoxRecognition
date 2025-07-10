from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, models, transforms

from pipeline import CLIPPredictor, ImageObjectPipeline


HOME = os.getcwd()
print(HOME)

# CLIP Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_predictor = CLIPPredictor(clip_model, clip_processor)

#ResNet50 load model
class_names = ['Statins', 'Nitrates', 'Heart-Failure-Medications', 'Antiplatelet-Agents', 
               'Antihypertensives', 'Anticoagulants', 'Antiarrhythmics', 'Combination-Medications', 'Other']

resnet50_predictor = ResNetPredictor(model_path='model_ft.pth', class_names=class_names, device='cuda')

yolo_path = os.path.join(HOME, 'pipeline/train17/weights/best.pt')

# Load the YOLOv8 model with your trained weights (weights.pt)
yolo_model = YOLO(yolo_path)  # Update with correct path to your .pt file

pipeline = ImageObjectPipeline(yolo_model, resnet50_predictor)

# Define your custom class names for zero-shot classification
custom_class_names = ['Statins', 'Nitrates', 'Heart-Failure-Medications', 'Antiplatelet-Agents', 
               'Antihypertensives', 'Anticoagulants', 'Antiarrhythmics', 'Combination-Medications', 'Other']

pipeline.process_image(os.path.join(HOME, "pipeline/sample/heartD/isosorbide.jpg"), custom_class_names)

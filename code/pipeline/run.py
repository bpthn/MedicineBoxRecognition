from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from pipeline import CLIPPredictor, ImageObjectPipeline


HOME = os.getcwd()
print(HOME)

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_predictor = CLIPPredictor(clip_model, clip_processor)

yolo_path = os.path.join(HOME, 'pipeline/train17/weights/best.pt')

# Load the YOLOv8 model with your trained weights (weights.pt)
yolo_model = YOLO(yolo_path)  # Update with correct path to your .pt file

pipeline = ImageObjectPipeline(yolo_model, clip_predictor)

# Define your custom class names for zero-shot classification
custom_class_names = ['Amiodarone', 'Amlodipine', 'Apixaban', 'Aspirin', 'Atorvastatin', 
                      'Caduet', 'Candesartan', 'Carvedilol', 'Clopidogrel', 'Dabigatran',
                      'Digoxin', 'Diltiazem', 'Eplerenone', 'Exforge', 'Furosemide', 
                      'Glyceryl trinitrate', 'Hydrochlorothiazide', 'Isosorbide mononitrate',
                      'Metoprolol', 'Perindopril', 'Rivaroxaban', 'Rosuvastatin', 'Simvastatin',
                      'Spironolactone', 'Statins', 'Ticagrelor', 'Valsartan', 'Warfarin', 'ramipril']

pipeline.process_image(os.path.join(HOME, "pipeline/sample/heartD/isosorbide.jpg"), custom_class_names)

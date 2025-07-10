from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, models, transforms
import torch.nn as nn

HOME = os.getcwd()

class BasePredictor:
    def predict(self, image, class_names):
        raise NotImplementedError


class CLIPPredictor(BasePredictor):
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor

    def predict(self, image, class_names):
        inputs = self.clip_processor(text=class_names, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image  # [1, num_classes]


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class ResNetPredictor:
    def __init__(self, model_path, class_names, device='cpu'):
        self.device = device
        self.class_names = class_names  # Store class names in the instance
        
        # Load base ResNet50 model
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        
        # Rebuild custom head used during training
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, len(class_names))  
        )
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 parameters
        for name, param in self.model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        
        # Unfreeze new fc parameters
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Same transforms as training (assumed default ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the same input size as training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):  # Only require the image (not class_names)
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform the prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
        
        # Get the predicted class
        confidence, predicted_class = torch.max(probs, 1)
        predicted_class = predicted_class.item()
        
        # Return the class probabilities, predicted index, and the predicted class name
        return probs, predicted_class, self.class_names[predicted_class]

class ImageObjectPipeline:
    def __init__(self, yolo_model, predictor, confidence_threshold=0.5):
        self.yolo_model = yolo_model
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

    def get_yolo_predictions(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(image_path)

        if len(results[0].boxes) == 0:
            print("No objects detected.")
            return [], img

        predictions = results[0].boxes.data.cpu().numpy()
        bboxes = predictions[:, :4]
        confidences = predictions[:, 4]
        valid_indices = confidences > self.confidence_threshold
        bboxes = bboxes[valid_indices]

        return bboxes, img

    def crop_image(self, image, bbox):
        x_min, y_min, x_max, y_max = map(int, bbox)
        pil_img = Image.fromarray(image)
        return pil_img.crop((x_min, y_min, x_max, y_max))

    def draw_box(self, image, bbox, label):
        draw = ImageDraw.Draw(image)
        x_min, y_min, x_max, y_max = map(int, bbox)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        # draw.text((x_min, y_min - 10), label, fill="red")
        return image

    def process_image(self, image_path, class_names):
        # Get YOLO predictions (bounding boxes and image)
        bboxes, img = self.get_yolo_predictions(image_path)
        pil_img = Image.fromarray(img)

        for i, bbox in enumerate(bboxes):
            # Crop image based on the bounding box
            cropped_img = self.crop_image(img, bbox)

            # Get the prediction from ResNetPredictor
            probs, predicted_idx, predicted_class = self.predictor.predict(cropped_img)  # Unpacking the 3 returned values
            
            # Get the predicted probability
            predicted_prob = probs[0][predicted_idx].item()  # Confidence of the predicted class

            print(f"bbox prediction from YOLOv8: {bbox}")
            print(f"Object {i+1}: Predicted class: {predicted_class} (confidence: {predicted_prob:.4f})")

            # Show cropped image with prediction
            plt.imshow(cropped_img)
            plt.title(f"Predicted: {predicted_class}")
            plt.axis("off")
            plt.show()

            # Draw bbox on original image
            self.draw_box(pil_img, bbox, predicted_class)

        # Show full image with all boxes and labels
        plt.figure(figsize=(10, 10))
        plt.imshow(pil_img)
        plt.title("All predictions on image")
        plt.axis("off")
        plt.show()




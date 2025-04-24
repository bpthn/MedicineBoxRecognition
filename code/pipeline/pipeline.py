from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

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
        draw.text((x_min, y_min - 10), label, fill="red")
        return image

    def process_image(self, image_path, class_names):
        bboxes, img = self.get_yolo_predictions(image_path)
        pil_img = Image.fromarray(img)

        for i, bbox in enumerate(bboxes):
            cropped_img = self.crop_image(img, bbox)
            probs = self.predictor.predict(cropped_img, class_names)  # shape: [1, num_classes]
            predicted_idx = probs.argmax(dim=1).item()
            predicted_class = class_names[predicted_idx]
            predicted_prob = probs[0][predicted_idx].item()

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

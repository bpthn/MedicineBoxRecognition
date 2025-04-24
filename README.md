# MedicineBoxRecognition

## Developing Object Recognition for Medicine Boxes

This project aims to develop an object recognition pipeline for medicine boxes using deep learning. The pipeline integrates both object detection and classification models to extract textual information from pharmaceutical packaging.

### Approach

In this work, we apply **YOLOv8** for **object detection** to identify and crop regions of interest (ROIs) that contain text from the packaging. Once the regions are identified, we compare two different approaches for text classification:

1. **Transfer Learning with ResNet**: A fine-tuned classification model based on ResNet is trained to recognise labels on the extracted regions.
2. **Zero-shot CLIP Model**: We use the CLIP model to predict the label by providing a textual prompt, allowing for zero-shot prediction without requiring additional training.

### Challenges Addressed

This system addresses common real-world challenges in medicine box recognition, such as:

- **Variable lighting**: The model adapts to different lighting conditions.
- **Cluttered backgrounds**: It effectively isolates the boxes from background noise.
- **Distorted text**: The system is trained to recognise text even when distorted or warped.



import os
from ultralytics import YOLO
import numpy as np
import cv2

class FundusYOLODetector:
    """
    YOLOv8 wrapper for detecting lesions or the Optic Disc in fundus images.
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        # If the specific eye-disease model doesn't exist, it defaults to the pre-trained COCO YOLOv8n
        # This allows the UI to run without crashing, but you will need to train a YOLO model on your data
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"Warning: {model_path} not found. Loading default YOLOv8n.")
            self.model = YOLO('yolov8n.pt')

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25):
        """
        Runs YOLO inference on the image and returns the plotted image with bounding boxes.
        
        Args:
            image (np.ndarray): The RGB image.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            np.ndarray: Image with bounding boxes drawn.
            list: List of detections (if needed for further processing).
        """
        # YOLOv8 expects BGR or RGB depending on how it's used, ultralytics handles RGB fine via PIL or numpy arrays
        results = self.model(image, conf=conf_threshold)
        
        # results[0].plot() returns a BGR image array with boxes
        plotted_image_bgr = results[0].plot()
        
        # Convert back to RGB for Streamlit/matplotlib
        plotted_image_rgb = cv2.cvtColor(plotted_image_bgr, cv2.COLOR_BGR2RGB)
        
        return plotted_image_rgb, results[0].boxes

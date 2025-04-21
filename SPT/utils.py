from torchvision import transforms
import numpy as np
import cv2
import torch
from ultralytics.engine.results import Results
from typing import Tuple

def process_for_yolo(img: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    """
    Preprocess an input img for YOLO model inference.
    
    Args:
        img: 16bit img
        target_size: Target size for resizing, default (640, 640)
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for YOLO inference
    """

    if np.max(img) > 2**16 - 1:
        img = cv2.normalize(
            src=img,
            dst=None,
            alpha=0,
            beta=2**16 - 1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_16U,
        )

    resizer = transforms.Resize(target_size)
    img = torch.from_numpy(img.astype(np.float32))
    img /= 2**16 - 1  # normalize
    img = img.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
    img = img.repeat(1, 3, 1, 1)  # repeat on channel dim
    img = resizer(img)  # Letter box preprocessing required (equiv to resize if square img)
    
    return img

def process_bounding_box(results: Results, frame_width: int, frame_height: int, model_size: Tuple[int, int] = (640, 640)):
    """
    Process a bounding box from model output to frame coordinates.

    Args:
        boxes: model output. 
        frame_width: int, frame width
        frame_height: int, frame height
        model_size: tuple, model size
    """
    processed_boxes = []
    for box in results.boxes:
        # Get the box coordinates
        reshaped_result = box.xywh.flatten()
        # Normalize to [0,1]
        reshaped_result = reshaped_result.cpu().numpy()
        reshaped_result /= np.array([model_size[0], model_size[1], model_size[0], model_size[1]])
        # Scale to frame dimensions
        reshaped_result *= np.array([frame_width, frame_height, frame_width, frame_height])
        reshaped_result = np.round(reshaped_result)
        
        # Convert from center x,y to top-left x,y
        x, y, w, h = reshaped_result
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Ensure coordinates are within frame boundaries
        x1 = max(0, min(x1, frame_width))
        y1 = max(0, min(y1, frame_height))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))
        
        # Get confidence and class as scalar values
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())
        
        # Add box coordinates, confidence, class, and detection index
        processed_boxes.append([x1, y1, x2, y2, conf, cls])
    
    return np.array(processed_boxes)
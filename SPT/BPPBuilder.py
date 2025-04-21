from ultralytics import YOLO
from .boosttrackpp import BoostTrackPP
import cv2
from .utils import process_bounding_box, process_for_yolo
from pathlib import Path
from typing import Union

class YOLOBPPBuilder:
    def __init__(self, model_path: Union[str, Path], reid_weights: Union[str, Path], confidence: float = 0.2, 
                 iou: float = 0.1,
                 max_det: int = 600,
                 device: str = "cpu",
                 half: bool = False):
        """
        Initialize the YOLO + BoostTrack++ tracker
        Args:
            model_path: str, path to the model
            reid_weights: str, path to the reid weights
            confidence: float, confidence threshold
            iou: float, iou threshold
            max_det: int, max detections
            device: str, device to run the model on
            half: bool, use half precision
        """
        self.model_path = model_path
        self.reid_weights = reid_weights
        self.device = device
        self.half = half
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.iou = iou
        self.max_det = max_det

    def load(self, path: Union[str, Path]):
        """
        Load a tif video to track
        """
        if not path.endswith(".tif") and not path.endswith(".tiff"):
            raise ValueError("File is not a tiff file")
        ret, mats = cv2.imreadmulti(
            path, [], flags=(cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        )
        if not ret:
            raise ValueError("opencv imreadmulti did not return an output")
        frames = []
        for img in mats:
            frames.append(img)
        self.video = frames


    def track(self, output_path: Union[str, Path]) -> None:
        """
        Track the video. Writes a csv file with the following track/bounding box info:
        
        frameno, id, x, y, w, h, score, class

        Args:
            output_path: str, path to save the csv file
        """
        bpp = BoostTrackPP(
            reid_weights=self.reid_weights,
            device=self.device,
            half=self.half
        )
        # create path to output if doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("frameno, id, x, y, w, h, score, class\n")
        for frame_no, frame in enumerate(self.video):
            img = process_for_yolo(frame)
            results = self.model(
                source = img, 
                conf=self.confidence, 
                iou=self.iou, 
                stream = False,
                max_det=self.max_det,
                device=self.device
            )
            boxes = process_bounding_box(results[0], frame.shape[1], frame.shape[0])
            # Convert processed image to numpy array for tracking
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tracks = bpp.update(boxes, img_np)
            for track in tracks:
                x, y, x2, y2, track_id, conf, cls, det_ind = track
                w = x2 - x
                h = y2 - y
                with open(output_path, "a") as f:
                    f.write(f"{frame_no}, {track_id}, {x}, {y}, {w}, {h}, {conf}, {cls}\n")
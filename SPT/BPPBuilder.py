from ultralytics import YOLO
from .boosttrackpp import BoostTrackPP
from .utils import process_bounding_box, process_for_yolo
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import yaml
from tracking import YoloLocalizer

from tracking import Stack, Frame, Loader, FileLoader
class YOLOBPPTracker:
    def __init__(self, model_path: Union[str, Path], reid_weights: Union[str, Path], 
                confidence: float = 0.1, iou: float = 0.5,
                max_det: int = 600,
                device: str = "cpu",
                half: bool = False,
                with_reid: bool = True,
                yolo_params_file: str = None,
                boost_params_file: str = None,
                loader: Loader = FileLoader(),
                localizer: YoloLocalizer = None):
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
            with_reid: bool, whether to use ReID
            yolo_params_file: str, path to YOLO parameters file
            boost_params_file: str, path to BoostTrack++ parameters file
            loader: Loader, loader to use for loading frames
        """
        self.model_path = model_path
        self.reid_weights = reid_weights
        self.device = device
        self.half = half
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.iou = iou
        self.max_det = max_det
        
        # Load YOLO parameters if provided
        if yolo_params_file is not None:    
            with open(yolo_params_file, "r") as f:
                yolo_params = yaml.load(f, Loader=yaml.FullLoader)
            self.confidence = yolo_params["confidence"]
            self.iou = yolo_params["iou"]
            self.max_det = yolo_params["max_det"]
        
        # Initialize default BoostTrack++ parameters
        boost_params = {
            "with_reid": with_reid
        }
        
        # Load BoostTrack++ parameters if provided
        if boost_params_file is not None:
            with open(boost_params_file, "r") as f:
                file_params = yaml.load(f, Loader=yaml.FullLoader)
                boost_params.update(file_params)
        
        # Initialize BoostTrack++ with parameters
        self.bpp = BoostTrackPP(
            reid_weights=self.reid_weights,
            device=self.device,
            half=self.half,
            **boost_params
        )
        
        self.loader = loader
        self.localizer = localizer
    def load(self, path: Union[str, Path]):
        """
        Load a tif video to track
        """
        if isinstance(path, Path):
            path = str(path)
        self.stack = self.loader(path)

    def track(self, frame_range: Optional[Tuple[int, int]] = None) -> None:
        """
        Track the video. Writes a csv file with the following track/bounding box info:
        
        frameno, id, x, y, w, h, score, class

        Args:
            output_path: str, path to save the csv file
        """
        
        # create path to output if doesn't exist
        frames = []
        for frame_no, frame in enumerate(self.stack):
            if frame_range is not None and (frame_no < frame_range[0] or frame_no > frame_range[1]):
                continue
            # Clear existing segments/tracks from the frame
            frame.segments = {}
            frame.confidences = {}
            frame.coords = {}
            frame.coord_data = {}
            img = process_for_yolo(frame.pixels)
            results = self.model(
                source = img, 
                conf=self.confidence, 
                iou=self.iou, 
                stream = False,
                max_det=self.max_det,
                device=self.device
            )
            boxes = process_bounding_box(results[0], frame.width, frame.height)
            # Convert processed image to numpy array for tracking
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

            tracks = self.bpp.update(boxes, img_np)

            for track in tracks:
                x, y, x2, y2, track_id, conf, cls, det_ind = track
                track_id = int(track_id)
                x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
                w = x2 - x
                h = y2 - y
                frame.segments[track_id] = (x, y, w, h)
                frame.confidences[track_id] = conf
                frame.coords[track_id] = (x + w/2, y + h/2) # temp
                frame.coord_data[track_id] = () # temp

            frames.append(frame)

    def localize(self):
        self.localizer(self.stack)
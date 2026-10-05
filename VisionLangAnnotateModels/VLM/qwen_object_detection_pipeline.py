import os
import json
import re
import time
import cv2
import datetime
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

# Traditional object detection imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection, AutoProcessor, AutoModelForObjectDetection
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Warning: DETR models not available. Install transformers for DETR support.")

try:
    from ensemble_boxes import nms, weighted_boxes_fusion
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("Warning: ensemble-boxes not available. Install with: pip install ensemble-boxes")

# SAM imports
try:
    from transformers import SamProcessor, SamModel
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM (Segment Anything Model) not available. Install transformers with SAM support for segmentation.")

# Import the VLM classifier
try:
    from vlm_classifierv4 import HuggingFaceVLM
except ImportError:
    # Try alternative import paths
    try:
        from .vlm_classifierv4 import HuggingFaceVLM
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from vlm_classifierv4 import HuggingFaceVLM

# Traditional Object Detector Classes
class YOLOv8Detector:
    """YOLOv8 object detector using Ultralytics."""
    
    def __init__(self, model_path="yolov8x.pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        self.model = YOLO(model_path)
    
    def detect(self, image):
        """Detect objects in image and return detections."""
        results = self.model(image)
        detections = []
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    cls = int(r.boxes.cls[i].item())
                    label = self.model.names[cls]
                    conf = r.boxes.conf[i].item()
                    detections.append({
                        "bbox": box.tolist(), 
                        "label": label, 
                        "confidence": conf
                    })
        return detections

class DETRDetector:
    """DETR object detector using Transformers."""
    
    def __init__(self, model_name="facebook/detr-resnet-50", threshold=0.3):
        if not DETR_AVAILABLE:
            raise ImportError("DETR models not available. Install transformers for DETR support.")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
    
    def detect(self, image):
        """Detect objects in image and return detections."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()  # [x0, y0, x1, y1]
            label_str = self.model.config.id2label[label.item()].lower()
            detections.append({
                "bbox": box,
                "label": label_str,
                "confidence": score.item()
            })
        return detections

class RTDETRDetector:
    """RT-DETR object detector using Transformers."""
    
    def __init__(self, model_name="SenseTime/deformable-detr", threshold=0.3):
        if not DETR_AVAILABLE:
            raise ImportError("DETR models not available. Install transformers for DETR support.")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
    
    def detect(self, image):
        """Detect objects in image and return detections."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_str = self.model.config.id2label[label.item()].lower()
            detections.append({
                "bbox": box.tolist(),
                "label": label_str,
                "confidence": score.item()
            })
        return detections

# Ensemble and Box Optimization Functions
def ensemble_detections(detection_lists, iou_thr=0.5, method='nms'):
    """Ensemble multiple detection results using NMS or WBF."""
    if not ENSEMBLE_AVAILABLE:
        print("Warning: ensemble-boxes not available. Using simple concatenation.")
        # Simple concatenation fallback
        all_detections = []
        for dets in detection_lists:
            all_detections.extend(dets)
        return all_detections
    
    if not detection_lists or all(len(dets) == 0 for dets in detection_lists):
        return []
    
    if method == 'wbf':
        return _wbf_ensemble_detections(detection_lists, iou_thr)
    else:
        return _nms_ensemble_detections(detection_lists, iou_thr)

def _nms_ensemble_detections(detection_lists, iou_thr=0.5):
    """Ensemble detections using Non-Maximum Suppression."""
    boxes, scores, labels = [], [], []
    for dets in detection_lists:
        for d in dets:
            boxes.append(d["bbox"])
            scores.append(d["confidence"])
            labels.append(d["label"])
    
    if not boxes:
        return []
    
    # Normalize boxes for ensemble-boxes (assumes max dimension of 1024)
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Get image dimensions from first box (rough estimate)
    max_coord = np.max(boxes)
    norm_factor = max(max_coord, 1024)
    
    boxes = boxes / norm_factor
    boxes_list = [boxes.tolist()]
    scores_list = [scores.tolist()]
    labels_list = [[label] for label in labels]
    
    ensembled_boxes, ensembled_scores, ensembled_labels = nms(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr
    )
    
    # Re-scale boxes
    ensembled_boxes = (np.array(ensembled_boxes) * norm_factor).tolist()
    return [
        {"bbox": b, "label": l, "confidence": s}
        for b, s, l in zip(ensembled_boxes, ensembled_scores, ensembled_labels)
    ]

def _wbf_ensemble_detections(detection_lists, iou_thr=0.5, skip_box_thr=0.3):
    """Ensemble detections using Weighted Boxes Fusion."""
    boxes_list, scores_list, labels_list = [], [], []
    
    # Get normalization factor from all boxes
    all_boxes = []
    for dets in detection_lists:
        for d in dets:
            all_boxes.extend(d["bbox"])
    
    if not all_boxes:
        return []
    
    norm_factor = max(max(all_boxes), 1024)
    
    for detections in detection_lists:
        boxes, scores, labels = [], [], []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([x1 / norm_factor, y1 / norm_factor, x2 / norm_factor, y2 / norm_factor])
            scores.append(d["confidence"])
            labels.append(d["label"])
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
    
    if not any(boxes_list):
        return []
    
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    results = []
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [coord * norm_factor for coord in b]
        results.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": s,
            "label": l
        })
    return results

def map_coco_to_allowed_classes(coco_class_name: str) -> str:
    """
    Map COCO class names to allowed detection classes.
    
    Args:
        coco_class_name: COCO class name (e.g., 'person', 'car', 'truck')
        
    Returns:
        Mapped class name from allowed_classes list
    """
    allowed_classes = [
        "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", "Pedestrian", "Worker", 
        "Street vendor", "Residential trash bin", "Commercial dumpster", "Street sign", "Construction sign", 
        "Traffic signal light", "Broken traffic lights", "Tree", "Overhanging branch", "Dumped trash", "Yard waste", 
        "Glass/debris", "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", "Cone", 
        "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", "Bicycle", "Scooter", "Wheelchair", 
        "Bus", "Train", "Ambulance", "Fire truck", "Other"
    ]
    
    # COCO to allowed classes mapping
    coco_mapping = {
        # Vehicles
        "car": "Car",
        "truck": "Truck",
        "bus": "Bus",
        "train": "Train",
        "motorcycle": "Other",
        "bicycle": "Bicycle",
        "airplane": "Other",
        "boat": "Other",
        
        # People
        "person": "Pedestrian",
        
        # Traffic infrastructure
        "traffic light": "Traffic signal light",
        "stop sign": "Street sign",
        "fire hydrant": "Other",
        "parking meter": "Other",
        
        # Street furniture
        "bench": "Bench",
        
        # Trees and vegetation
        "potted plant": "Tree",
        
        # Animals -> Other
        "bird": "Other",
        "cat": "Other",
        "dog": "Other",
        "horse": "Other",
        "sheep": "Other",
        "cow": "Other",
        "elephant": "Other",
        "bear": "Other",
        "zebra": "Other",
        "giraffe": "Other",
        
        # Objects that could be trash/debris
        "bottle": "Glass/debris",
        "cup": "Glass/debris",
        "banana": "Dumped trash",
        "apple": "Dumped trash",
        "sandwich": "Dumped trash",
        "orange": "Dumped trash",
        "pizza": "Dumped trash",
        "donut": "Dumped trash",
        "cake": "Dumped trash",
    }
    
    # Return mapped class or 'Other' if not found
    return coco_mapping.get(coco_class_name.lower(), "Other")


def optimize_boxes_for_vlm(detections, min_box_size=30, merge_threshold=0.3, max_boxes=20):
    """Optimize bounding boxes for VLM processing by merging small/nearby boxes."""
    if not detections:
        return detections
    
    # Filter out very small boxes (less aggressive filtering)
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        width, height = x2 - x1, y2 - y1
        # Keep boxes that meet minimum size requirements (OR instead of AND for less aggressive filtering)
        if width >= min_box_size or height >= min_box_size:
            filtered_detections.append(det)
    
    if len(filtered_detections) <= max_boxes:
        return filtered_detections
    
    # Sort by confidence and take top boxes
    filtered_detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Try to merge nearby boxes
    merged_detections = []
    used_indices = set()
    
    for i, det1 in enumerate(filtered_detections):
        if i in used_indices:
            continue
        
        # Find nearby boxes to merge
        boxes_to_merge = [det1]
        used_indices.add(i)
        
        for j, det2 in enumerate(filtered_detections[i+1:], i+1):
            if j in used_indices:
                continue
            
            if _should_merge_boxes(det1["bbox"], det2["bbox"], merge_threshold):
                boxes_to_merge.append(det2)
                used_indices.add(j)
        
        # Merge boxes if multiple found
        if len(boxes_to_merge) > 1:
            merged_box = _merge_multiple_boxes(boxes_to_merge)
            merged_detections.append(merged_box)
        else:
            merged_detections.append(det1)
        
        if len(merged_detections) >= max_boxes:
            break
    
    return merged_detections[:max_boxes]

def _should_merge_boxes(bbox1, bbox2, threshold=0.3):
    """Check if two boxes should be merged based on IoU or proximity."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate IoU
    intersection_x1 = max(x1_1, x1_2)
    intersection_y1 = max(y1_1, y1_2)
    intersection_x2 = min(x2_1, x2_2)
    intersection_y2 = min(y2_1, y2_2)
    
    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        intersection_area = 0
    else:
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # Also check proximity (boxes that are very close)
    center1_x, center1_y = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    center2_x, center2_y = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    
    avg_width = ((x2_1 - x1_1) + (x2_2 - x1_2)) / 2
    avg_height = ((y2_1 - y1_1) + (y2_2 - y1_2)) / 2
    
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    proximity_threshold = min(avg_width, avg_height) * 0.5
    
    return iou > threshold or distance < proximity_threshold

def _merge_multiple_boxes(detections):
    """Merge multiple detection boxes into one."""
    if len(detections) == 1:
        return detections[0]
    
    # Find bounding box that encompasses all
    x1_min = min(det["bbox"][0] for det in detections)
    y1_min = min(det["bbox"][1] for det in detections)
    x2_max = max(det["bbox"][2] for det in detections)
    y2_max = max(det["bbox"][3] for det in detections)
    
    # Use highest confidence and combine labels
    max_conf_det = max(detections, key=lambda x: x["confidence"])
    labels = [det["label"] for det in detections]
    unique_labels = list(set(labels))
    
    return {
        "bbox": [x1_min, y1_min, x2_max, y2_max],
        "label": "/".join(unique_labels),
        "confidence": max_conf_det["confidence"]
    }


def ensemble_hybrid_vlm_detections(traditional_detections, vlm_detections, 
                                 ensemble_method='nms', box_merge_threshold=0.3,
                                 matching_method='iou', iou_threshold=0.3, overlap_threshold=0.5):
    """
    Ensemble traditional detector and VLM detections with hybrid logic.
    
    Args:
        traditional_detections: List of traditional detector results
        vlm_detections: List of VLM detection results
        ensemble_method: Method for ensembling ('nms' or 'wbf')
        box_merge_threshold: Threshold for merging nearby boxes
        matching_method: Method for matching VLM to traditional detections ('iou' or 'overlap')
        iou_threshold: IoU threshold for matching (when matching_method='iou')
        overlap_threshold: Overlap threshold for matching (when matching_method='overlap')
        
    Returns:
        List of final hybrid detection objects
    """
    all_objects = []
    
    # Step 1: Ensemble all detections (traditional + VLM)
    all_detection_lists = []
    if traditional_detections:
        all_detection_lists.extend(traditional_detections)
    if vlm_detections:
        all_detection_lists.append(vlm_detections)
        
    if not all_detection_lists:
        return all_objects
        
    print(f"Ensembling {len(all_detection_lists)} detection sources...")
    ensembled_detections = ensemble_detections(all_detection_lists, method=ensemble_method)
    
    # Step 2: Optimize boxes (merge small/nearby boxes)
    if ensembled_detections:
        print(f"Optimizing {len(ensembled_detections)} boxes...")
        optimized_detections = optimize_boxes_for_vlm(ensembled_detections, merge_threshold=box_merge_threshold)
        print(f"Final ensemble result: {len(optimized_detections)} objects")
        
        # Step 3: Convert to final format with hybrid logic:
        # - Prioritize VLM object names and descriptions
        # - Use traditional detector bounding boxes when available
        # - For objects not detected by VLM, use original COCO class with no description
        for det in optimized_detections:
            # Check if this detection has VLM information
            vlm_match = None
            for vlm_det in vlm_detections:
                # Check if VLM detection matches with this detection
                if matching_method == 'iou':
                    match_score = _calculate_iou_static(det['bbox'], vlm_det['bbox'])
                    threshold = iou_threshold
                else:  # overlap method
                    match_score = _calculate_overlap_ratio(det['bbox'], vlm_det['bbox'])
                    threshold = overlap_threshold
                    
                if match_score > threshold:
                    vlm_match = vlm_det
                    break
            
            #if vlm_match and det.get('source') == 'vlm':
            if vlm_match and vlm_match.get('source') == 'vlm':
                # VLM detection - use VLM name and description
                all_objects.append({
                    'label': str(vlm_match['label']) if isinstance(vlm_match['label'], (np.str_, np.bytes_)) else vlm_match['label'],
                    'bbox': det['bbox'],  # Use ensembled bbox (may be from traditional detector)
                    'description': vlm_match.get('description', f"Detected by VLM: {vlm_match['label']}"),
                    'confidence': det['confidence'],
                    'source': 'hybrid_vlm'
                })
            elif det.get('original_coco_label'):
                # Traditional detector only - use mapped class with no description
                all_objects.append({
                    'label': str(det['label']) if isinstance(det['label'], (np.str_, np.bytes_)) else det['label'],  # Already mapped to allowed classes
                    'bbox': det['bbox'],
                    'description': f"Detected by traditional detector: {det['original_coco_label']} -> {det['label']}",
                    'confidence': det['confidence'],
                    'source': 'hybrid_traditional'
                })
            else:
                # Fallback for other cases
                all_objects.append({
                    'label': str(det['label']) if isinstance(det['label'], (np.str_, np.bytes_)) else det['label'],
                    'bbox': det['bbox'],
                    'description': f"Detected by ensemble: {det['label']}",
                    'confidence': det['confidence'],
                    'source': det.get('source', 'ensemble')
                })
    
    return all_objects


def _calculate_iou_static(bbox1, bbox2):
    """
    Static version of IoU calculation for use in ensemble function.
    
    Args:
        bbox1: [x1, y1, x2, y2] format
        bbox2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _calculate_overlap_ratio(bbox1, bbox2):
    """
    Calculate overlap ratio between two bounding boxes.
    This is more lenient than IoU and focuses on how much the smaller box overlaps with the larger one.
    
    Args:
        bbox1: [x1, y1, x2, y2] format
        bbox2: [x1, y1, x2, y2] format
        
    Returns:
        Overlap ratio between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Use the smaller area as denominator for more lenient matching
    smaller_area = min(area1, area2)
    
    return intersection / smaller_area if smaller_area > 0 else 0.0


class QwenObjectDetectionPipeline:
    """
    A robust object detection pipeline using Qwen2.5-VL model.
    
    This pipeline:
    1. Uses Qwen2.5-VL to detect objects in images
    2. Parses the natural language response to extract bounding box coordinates
    3. Converts results to Label Studio compatible JSON format
    4. Visualizes detection results with bounding boxes overlaid on images
    5. Saves all results (JSON and visualizations) to organized output directories
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda",
                 output_dir: str = "./object_detection_results",
                 enable_sam: bool = False,
                 enable_traditional_detectors: bool = False,
                 traditional_detectors: List[str] = None):
        """Initialize the object detection pipeline.
        
        Args:
            model_name: Qwen2.5-VL model name
            device: Device to run the model on (cuda/cpu)
            output_dir: Directory to save results
            enable_sam: Whether to enable SAM segmentation capabilities
            enable_traditional_detectors: Whether to enable traditional object detectors
            traditional_detectors: List of traditional detectors to use ['yolo', 'detr', 'rtdetr']
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.enable_sam = enable_sam and SAM_AVAILABLE
        self.enable_traditional_detectors = enable_traditional_detectors
        
        # Initialize allowed classes for object detection
        # self.allowed_classes = [
        #     "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrian", "Person",
        #     "Traffic_light", "Traffic_sign", "Stop_sign", "Parking_meter",
        #     "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant",
        #     "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag",
        #     "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports_ball",
        #     "Kite", "Baseball_bat", "Baseball_glove", "Skateboard", "Surfboard",
        #     "Tennis_racket", "Bottle", "Wine_glass", "Cup", "Fork", "Knife",
        #     "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
        #     "Carrot", "Hot_dog", "Pizza", "Donut", "Cake", "Chair", "Couch",
        #     "Potted_plant", "Bed", "Dining_table", "Toilet", "TV", "Laptop",
        #     "Mouse", "Remote", "Keyboard", "Cell_phone", "Microwave", "Oven",
        #     "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase",
        #     "Scissors", "Teddy_bear", "Hair_drier", "Toothbrush", "Other"
        # ]

                # Define allowed classes for consistent detection
        self.allowed_classes = [
            "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", "Pedestrian", "Worker", 
            "Street vendor", "Residential trash bin", "Commercial dumpster", "Street sign", "Construction sign", 
            "Traffic signal light", "Broken traffic lights", "Tree", "Overhanging branch", "Dumped trash", "Yard waste", 
            "Glass/debris", "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", "Cone", 
            "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", "Bicycle", "Scooter", "Wheelchair", 
            "Bus", "Train", "Ambulance", "Fire truck", "Other"
        ]

        # Object detection prompt template
        
        self.detection_prompt = (
            "Detect objects in this image and provide ONLY the structured output format below. "
            "Do NOT provide explanations, analysis, or verbose descriptions.\n\n"
            "REQUIRED FORMAT:\n"
            "RESULT_START\n"
            "Object_name: (x1,y1,x2,y2) confidence description\n"
            "Object_name: (x1,y1,x2,y2) confidence description\n"
            "RESULT_END\n\n"
            "EXAMPLES:\n"
            "RESULT_START\n"
            "Car: (100,50,200,150) 0.95 red sedan parked\n"
            "Tree: (300,20,350,180) 0.88 large oak tree\n"
            "Pedestrian: (150,80,180,200) 0.92 person walking\n"
            "RESULT_END\n\n"
            "ALLOWED CLASSES (use ONLY these exact names):\n"
            f"{', '.join(self.allowed_classes)}\n\n"
            "IMPORTANT: Use ONLY the allowed class names listed above. If an object doesn't match any allowed class, use 'Other'.\n"
            "IMPORTANT: If no clear objects are visible or image is too blurry, respond with: 'RESULT_START\\nNo objects detected\\nRESULT_END'\n"
            "IMPORTANT: Always wrap your response with RESULT_START and RESULT_END markers. Use ONLY the exact format shown above."
        )
        
        # Initialize the VLM model
        print(f"Initializing {model_name}...")
        self.vlm = HuggingFaceVLM(model_name=model_name, device=device)
        
        # Initialize traditional detectors if enabled
        self.traditional_detectors = []
        if self.enable_traditional_detectors and traditional_detectors:
            self._initialize_traditional_detectors(traditional_detectors)
        
        # Initialize SAM if enabled
        self.sam_processor = None
        self.sam_model = None
        if self.enable_sam:
            try:
                print("Loading Segment Anything Model...")
                self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
                print("SAM model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load SAM model: {e}")
                self.enable_sam = False
        

        
        # Create output directories
        self._setup_output_directories()
    
    def _initialize_traditional_detectors(self, detector_names: List[str]):
        """Initialize traditional object detectors."""
        for detector_name in detector_names:
            try:
                if detector_name.lower() == 'yolo':
                    if YOLO_AVAILABLE:
                        detector = YOLOv8Detector(model_path="yolov8x.pt")
                        self.traditional_detectors.append(detector)
                        print(f"Initialized YOLOv8 detector")
                    else:
                        print(f"Warning: YOLO not available, skipping")
                
                elif detector_name.lower() == 'detr':
                    if DETR_AVAILABLE:
                        detector = DETRDetector(model_name="facebook/detr-resnet-50")
                        self.traditional_detectors.append(detector)
                        print(f"Initialized DETR detector")
                    else:
                        print(f"Warning: DETR not available, skipping")
                
                elif detector_name.lower() == 'rtdetr':
                    if DETR_AVAILABLE:
                        detector = RTDETRDetector(model_name="SenseTime/deformable-detr")
                        self.traditional_detectors.append(detector)
                        print(f"Initialized RT-DETR detector")
                    else:
                        print(f"Warning: RT-DETR not available, skipping")
                
                else:
                    print(f"Warning: Unknown detector '{detector_name}', skipping")
            
            except Exception as e:
                print(f"Warning: Failed to initialize {detector_name} detector: {e}")
        
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        self.json_dir = os.path.join(self.output_dir, "json_annotations")
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        self.raw_dir = os.path.join(self.output_dir, "raw_responses")
        self.seg_dir = os.path.join(self.output_dir, "segmentations")
        
        for dir_path in [self.json_dir, self.viz_dir, self.raw_dir, self.seg_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Generate a timestamp string for file naming."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_bbox_specific_prompt(self, bounding_boxes: List[Tuple[int, int, int, int]]) -> str:
        """
        Generate a prompt for VLM to analyze specific bounding boxes in the image.
        
        Args:
            bounding_boxes: List of bounding boxes as (x1, y1, x2, y2) tuples
            
        Returns:
            Formatted prompt string for VLM analysis
        """
        if not bounding_boxes:
            return self.detection_prompt
        
        bbox_strings = []
        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            bbox_strings.append(f"({x1}, {y1}, {x2}, {y2})")
        
        bbox_list = ", ".join(bbox_strings)
        
        prompt = (
            f"Analyze the objects inside these specific bounding boxes in the image: {bbox_list}\n\n"
            "For each bounding box, identify what object is inside it. If a box contains no clear object or is empty, respond with 'Empty' for that box.\n\n"
            "REQUIRED FORMAT (one result per bounding box, in the same order as given):\n"
            "RESULT_START\n"
            "BOX_INDEX_0: Object_name confidence description\n"
            "BOX_INDEX_1: Object_name confidence description\n"
            "BOX_INDEX_2: Object_name confidence description\n"
            "RESULT_END\n\n"
            "EXAMPLES:\n"
            "RESULT_START\n"
            "BOX_INDEX_0: Car 0.95 red sedan parked\n"
            "BOX_INDEX_1: Tree 0.88 large oak tree\n"
            "BOX_INDEX_2: Empty 0.0 no clear object visible\n"
            "RESULT_END\n\n"
            "ALLOWED CLASSES (use ONLY these exact names):\n"
            f"{', '.join(self.allowed_classes)}\n\n"
            "IMPORTANT: Use ONLY the allowed class names listed above. If an object doesn't match any allowed class, use 'Other'.\n"
            "IMPORTANT: If a bounding box contains no clear object, use 'Empty' as the class name.\n"
            "IMPORTANT: Analyze each bounding box in the exact order provided (BOX_INDEX_0, BOX_INDEX_1, etc.).\n"
            "IMPORTANT: Use ONLY the exact format shown above with RESULT_START and RESULT_END markers.\n"
            "IMPORTANT: Do NOT include coordinate numbers in your response, only use BOX_INDEX_N format."
        )
        
        return prompt
    
    def _parse_bbox_specific_response(self, response: str, original_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Parse VLM response for bounding box-specific analysis.
        
        Args:
            response: Raw VLM response text
            original_boxes: Original bounding boxes that were analyzed
            
        Returns:
            List of detection dictionaries with VLM analysis results
        """
        detections = []
        
        if not response or not original_boxes:
            return detections
        
        # Try new structured format first
        if "RESULT_START" in response and "RESULT_END" in response:
            return self._parse_structured_response(response, original_boxes)
        
        # Fallback to old format parsing for backward compatibility
        return self._parse_legacy_response(response, original_boxes)
    
    def _parse_structured_response(self, response: str, original_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Parse structured VLM response with BOX_INDEX format.
        """
        detections = []
        
        # Extract content between RESULT_START and RESULT_END
        start_idx = response.find("RESULT_START")
        end_idx = response.find("RESULT_END")
        
        if start_idx == -1 or end_idx == -1:
            print("Warning: Could not find RESULT_START/RESULT_END markers")
            return self._parse_legacy_response(response, original_boxes)
        
        result_content = response[start_idx + len("RESULT_START"):end_idx].strip()
        lines = [line.strip() for line in result_content.split('\n') if line.strip()]
        
        # Parse each BOX_INDEX line
        for i, (x1, y1, x2, y2) in enumerate(original_boxes):
            box_index_pattern = f"BOX_INDEX_{i}:"
            found_analysis = False
            
            for line in lines:
                if line.startswith(box_index_pattern):
                    # Extract the analysis part after BOX_INDEX_N:
                    analysis_part = line.split(':', 1)[1].strip()
                    
                    # Parse: Object_name confidence description
                    parts = analysis_part.split(' ', 2)
                    if len(parts) >= 2:
                        label = parts[0]
                        try:
                            confidence = float(parts[1])
                        except (ValueError, IndexError):
                            confidence = 0.8
                        
                        description = parts[2] if len(parts) > 2 else f"Detected by VLM: {label}"
                        
                        # Skip empty boxes
                        if label.lower() != 'empty':
                            # Map to allowed classes
                            mapped_label = self._map_vlm_to_allowed_classes(label)
                            
                            detections.append({
                                'label': mapped_label,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'description': description,
                                'source': 'vlm_sequential'
                            })
                    
                    found_analysis = True
                    break
            
            # If no analysis found for this box, skip it (considered empty)
            if not found_analysis:
                print(f"Warning: No VLM analysis found for BOX_INDEX_{i} (box {x1},{y1},{x2},{y2})")
        
        return detections
    
    def _parse_legacy_response(self, response: str, original_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Parse legacy VLM response format with coordinate matching (more robust).
        """
        detections = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Use regex to extract box information more robustly
        import re
        box_pattern = re.compile(r'Box\s*\(([^)]+)\):\s*(.+)')
        
        parsed_boxes = []
        for line in lines:
            match = box_pattern.match(line)
            if match:
                coords_str = match.group(1)
                analysis = match.group(2).strip()
                
                # Try to extract coordinates (handle both int and float)
                try:
                    coords = [float(x.strip()) for x in coords_str.split(',')]
                    if len(coords) == 4:
                        parsed_boxes.append((coords, analysis))
                except ValueError:
                    continue
        
        # Match parsed boxes to original boxes by finding closest matches
        for x1, y1, x2, y2 in original_boxes:
            best_match = None
            min_distance = float('inf')
            
            for coords, analysis in parsed_boxes:
                px1, py1, px2, py2 = coords
                # Calculate distance between box centers
                orig_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                parsed_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                distance = ((orig_center[0] - parsed_center[0]) ** 2 + (orig_center[1] - parsed_center[1]) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = analysis
            
            if best_match and min_distance < 50:  # Reasonable threshold for matching
                # Parse: Object_name confidence description
                parts = best_match.split(' ', 2)
                if len(parts) >= 2:
                    label = parts[0]
                    try:
                        confidence = float(parts[1])
                    except (ValueError, IndexError):
                        confidence = 0.8
                    
                    description = parts[2] if len(parts) > 2 else f"Detected by VLM: {label}"
                    
                    # Skip empty boxes
                    if label.lower() != 'empty':
                        # Map to allowed classes
                        mapped_label = self._map_vlm_to_allowed_classes(label)
                        
                        detections.append({
                            'label': mapped_label,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'description': description,
                            'source': 'vlm_sequential'
                        })
        
        return detections
    
    def _map_vlm_to_allowed_classes(self, vlm_class_name: str) -> str:
        """
        Map VLM detected class names to allowed detection classes.
        
        Args:
            vlm_class_name: Class name detected by VLM
            
        Returns:
            Mapped class name from allowed_classes list
        """
        vlm_class_lower = vlm_class_name.lower().strip()
        
        # Direct matches (case insensitive)
        for allowed_class in self.allowed_classes:
            if vlm_class_lower == allowed_class.lower():
                return allowed_class
        
        # Fuzzy matching for common variations
        mapping = {
            'person': 'Pedestrian',
            'people': 'Pedestrian', 
            'human': 'Pedestrian',
            'man': 'Pedestrian',
            'woman': 'Pedestrian',
            'vehicle': 'Car',
            'automobile': 'Car',
            'sedan': 'Car',
            'suv': 'Car',
            'van': 'Truck',
            'pickup': 'Truck',
            'lorry': 'Truck',
            'bike': 'Bicycle',
            'motorcycle': 'Scooter',
            'motorbike': 'Scooter',
            'sign': 'Street sign',
            'traffic light': 'Traffic signal light',
            'stoplight': 'Traffic signal light',
            'signal': 'Traffic signal light',
            'pole': 'Utility pole',
            'post': 'Utility pole',
            'trash': 'Dumped trash',
            'garbage': 'Dumped trash',
            'waste': 'Dumped trash',
            'rubbish': 'Dumped trash',
            'debris': 'Glass/debris',
            'construction worker': 'Worker',
            'road worker': 'Worker',
            'maintenance worker': 'Worker'
        }
        
        # Check fuzzy mappings
        for key, value in mapping.items():
            if key in vlm_class_lower:
                return value
        
        # Default to 'Other' if no match found
        return 'Other'
    
    def _parse_detection_response(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Robust parser for Qwen2.5-VL's natural language response to extract object detection results.
        Handles multiple response formats including structured format with markers and legacy formats.
        
        Args:
            response: Raw response from Qwen2.5-VL
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            List of detected objects with bounding boxes
        """
        objects = []
        
        # Clean up the response
        response = response.strip()
        
        # Method 1: Try new structured format with RESULT_START/RESULT_END markers
        if "RESULT_START" in response and "RESULT_END" in response:
            objects = self._parse_structured_detection_response(response, image_width, image_height)
            if objects is not None:  # None means parsing failed, empty list means no objects
                return objects
        
        # Method 2: Try simple structured format (Object: (x,y,x,y) confidence description)
        objects = self._parse_simple_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 3: Try numbered format (1. **Object** ... **Bounding Box:** (x,y,x,y))
        objects = self._parse_numbered_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 4: Try general pattern matching for any coordinate format
        objects = self._parse_general_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 5: Handle verbose text responses with no structured data
        # Check if response contains analysis but no coordinates
        if self._is_verbose_analysis(response):
            print("Warning: VLM returned verbose analysis without structured detection data")
            return []  # Return empty list for verbose responses without coordinates
        
        print(f"Warning: Could not parse detection response: {response[:100]}...")
        return objects
    
    def _parse_structured_detection_response(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse structured format with RESULT_START/RESULT_END markers for main detection responses.
        
        Args:
            response: Raw VLM response text
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            List of detection dictionaries, or None if parsing failed
        """
        try:
            # Extract content between RESULT_START and RESULT_END
            start_marker = "RESULT_START"
            end_marker = "RESULT_END"
            
            start_idx = response.find(start_marker)
            end_idx = response.find(end_marker)
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                return None
            
            # Extract the content between markers
            content = response[start_idx + len(start_marker):end_idx].strip()
            
            # Handle "No objects detected" case
            if "no objects detected" in content.lower():
                return []
            
            detections = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: Object_name: (x1,y1,x2,y2) confidence description
                match = re.match(r'^([^:]+):\s*\((\d+),(\d+),(\d+),(\d+)\)\s*([\d.]+)?\s*(.*)$', line)
                if match:
                    label = match.group(1).strip()
                    x1, y1, x2, y2 = map(int, match.groups()[1:5])
                    confidence = float(match.group(6)) if match.group(6) else 0.8
                    description = match.group(7).strip() if match.group(7) else f"Detected {label.lower()}"
                    
                    # Validate bounding box
                    if self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                        # Map VLM detected class to allowed classes
                        mapped_label = self._map_vlm_to_allowed_classes(label)
                        
                        detections.append({
                            'label': mapped_label,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'description': description,
                            'original_vlm_label': label
                        })
                    else:
                        print(f"Warning: Invalid bounding box for {label}: ({x1},{y1},{x2},{y2})")
            
            return detections
            
        except Exception as e:
            print(f"Error parsing structured detection response: {e}")
            return None
    
    def _parse_simple_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse simple format: Object: (x,y,x,y) confidence description
        """
        objects = []
        
        # Check for explicit 'no objects' response
        if 'no objects detected' in response.lower():
            return objects  # Return empty list
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Pattern: Object: (x,y,x,y) confidence description
            match = re.match(r'^([^:]+):\s*\((\d+),(\d+),(\d+),(\d+)\)\s*([\d.]+)?\s*(.*)$', line)
            if match:
                label = match.group(1).strip()
                x1, y1, x2, y2 = map(int, match.groups()[1:5])
                confidence = float(match.group(6)) if match.group(6) else 1.0
                description = match.group(7).strip() if match.group(7) else f"A {label.lower()} detected"
                
                if self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                    # Map VLM detected class to allowed classes
                    mapped_label = self._map_vlm_to_allowed_classes(label)
                    objects.append({
                        'label': mapped_label,
                        'bbox': [x1, y1, x2, y2],
                        'description': description,
                        'confidence': confidence,
                        'original_vlm_label': label  # Keep original for reference
                    })
        
        return objects
    
    def _parse_numbered_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse numbered format: 1. **Object** ... **Bounding Box:** (x,y,x,y)
        """
        objects = []
        
        # Split response into numbered sections
        sections = re.split(r'\n(?=\d+\.\s*\*\*)', response)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Extract object label from numbered format
            label_match = re.search(r'\d+\.\s*\*\*([^*]+)\*\*', section)
            if not label_match:
                continue
                
            object_label = label_match.group(1).strip()
            
            # Extract bounding box coordinates
            bbox_patterns = [
                r'\*\*Bounding Box Coordinates\*\*:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
                r'\*\*Bounding Box:\*\*\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
                r'Bounding Box:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
            ]
            
            for pattern in bbox_patterns:
                bbox_match = re.search(pattern, section)
                if bbox_match:
                    try:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        
                        if self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                            # Extract description
                            description_match = re.search(r'Description:\s*(.+?)(?=\n\n|\n\d+\.|$)', section, re.DOTALL)
                            description = description_match.group(1).strip() if description_match else f"A {object_label.lower()} detected in the image"
                            
                            # Map VLM detected class to allowed classes
                            mapped_label = self._map_vlm_to_allowed_classes(object_label)
                            
                            objects.append({
                                'label': mapped_label,
                                'bbox': [x1, y1, x2, y2],
                                'description': description,
                                'confidence': 1.0,
                                'original_vlm_label': object_label  # Keep original for reference
                            })
                            break
                    except (ValueError, IndexError):
                        continue
        
        return objects
    
    def _parse_general_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse general format using pattern matching for coordinates
        """
        objects = []
        
        # Find all coordinate patterns in the response
        coord_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        coord_matches = list(re.finditer(coord_pattern, response))
        
        if not coord_matches:
            return objects
        
        # Split response into lines and try to match coordinates with labels
        lines = response.split('\n')
        
        for match in coord_matches:
            x1, y1, x2, y2 = map(int, match.groups())
            
            if not self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                continue
            
            # Find the line containing this coordinate
            coord_text = match.group(0)
            matching_line = None
            
            for line in lines:
                if coord_text in line:
                    matching_line = line
                    break
            
            if matching_line:
                # Extract label from the line
                label = self._extract_object_label(matching_line)
                description = self._extract_description(matching_line)
                
                # Map VLM detected class to allowed classes
                mapped_label = self._map_vlm_to_allowed_classes(label)
                
                objects.append({
                    'label': mapped_label,
                    'bbox': [x1, y1, x2, y2],
                    'description': description,
                    'confidence': 1.0,
                    'original_vlm_label': label  # Keep original for reference
                })
        
        return objects
    
    def _is_verbose_analysis(self, response: str) -> bool:
        """
        Check if response is a verbose analysis without structured detection data
        """
        response_lower = response.lower()
        
        # Check for explicit 'no objects detected' response (this is valid, not verbose)
        if 'no objects detected' in response_lower:
            return False
        
        # Look for indicators of verbose analysis
        verbose_indicators = [
            'the provided image',
            'based on the analysis',
            'however,',
            'there are no',
            'no distinct',
            'no discernible',
            'no clear',
            'too blurry',
            'lacks clear details',
            'appears to be'
        ]
        
        # Check if response contains verbose indicators and no coordinates
        has_verbose_indicators = any(indicator in response_lower for indicator in verbose_indicators)
        has_coordinates = bool(re.search(r'\(\d+,\s*\d+,\s*\d+,\s*\d+\)', response))
        
        return has_verbose_indicators and not has_coordinates
    
    def _extract_object_label(self, section: str) -> str:
        """
        Extract object label from a section of text.
        
        Args:
            section: Text section containing object information
            
        Returns:
            Extracted object label
        """
        # Try to find labels in various formats
        label_patterns = [
            r'\*\*([^*]+)\*\*',  # **Label**
            r'(\d+\.\s*\*\*[^*]+\*\*)',  # 1. **Label**
            r'(\d+\.\s*[^:]+):',  # 1. Label:
            r'^([^:]+):',  # Label: at start of line
            r'(\w+(?:\s+\w+)*?)(?:\s*\(|\s*:)',  # Word(s) before ( or :
        ]
        
        for pattern in label_patterns:
            match = re.search(pattern, section, re.MULTILINE)
            if match:
                label = match.group(1).strip()
                # Clean up the label
                label = re.sub(r'^\d+\.\s*', '', label)  # Remove numbering
                label = re.sub(r'\*\*', '', label)  # Remove markdown
                label = label.strip()
                if label and len(label) > 0:
                    return label
        
        # Fallback: use first few words
        words = section.split()[:3]
        return ' '.join(words) if words else 'Unknown Object'
    
    def _extract_description(self, section: str) -> str:
        """
        Extract object description from a section of text.
        
        Args:
            section: Text section containing object information
            
        Returns:
            Extracted description
        """
        # Look for description patterns
        desc_patterns = [
            r'\*\*Description:\*\*\s*(.+?)(?=\n|$)',
            r'Description:\s*(.+?)(?=\n|$)',
            r'\*\*Bounding Box:\*\*.*?\n(.+?)(?=\n|$)',
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, section, re.DOTALL)
            if match:
                desc = match.group(1).strip()
                if desc:
                    return desc
        
        # Fallback: use the section text, cleaned up
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\d+\.|\*\*|Bounding Box:', line):
                return line
        
        return "Object detected in image"
    
    def _validate_bbox(self, x1: int, y1: int, x2: int, y2: int, 
                      image_width: int, image_height: int) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            image_width, image_height: Image dimensions
            
        Returns:
            True if bbox is valid, False otherwise
        """
        # Check if coordinates are within image bounds
        if x1 < 0 or y1 < 0 or x2 >= image_width or y2 >= image_height:
            return False
        
        # Check if x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Check if bbox has reasonable size (at least 5x5 pixels)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return False
        
        # Filter out unreasonably large bounding boxes
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        
        # Reject bboxes that cover more than 80% of the image
        if bbox_area > (0.8 * image_area):
            return False
        
        # Reject bboxes that are too wide or too tall relative to image
        if bbox_width > (0.9 * image_width) or bbox_height > (0.9 * image_height):
            return False
        
        # Reject extremely thin or narrow bboxes (aspect ratio checks)
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        return True
    
    def _blur_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, blur_strength: int = 15) -> np.ndarray:
        """
        Apply Gaussian blur to a specific region of the image.
        
        Args:
            image: Input image as numpy array
            x1, y1, x2, y2: Bounding box coordinates of region to blur
            blur_strength: Strength of the blur effect (higher = more blur)
            
        Returns:
            np.ndarray: Image with blurred region
        """
        # Create a copy of the image
        blurred_image = image.copy()
        
        # Extract the region to blur
        region = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur to the region
        blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
        
        # Replace the original region with the blurred version
        blurred_image[y1:y2, x1:x2] = blurred_region
        
        return blurred_image
    
    def _normalize_bbox(self, bbox):
        """
        Normalize bbox to dictionary format from either list or dict format.
        
        Args:
            bbox: Either [x1, y1, x2, y2] list or {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} dict
            
        Returns:
            Dictionary with keys x1, y1, x2, y2
        """
        if isinstance(bbox, list) and len(bbox) == 4:
            return {'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]}
        elif isinstance(bbox, dict):
            return bbox
        else:
            return {}
    
    def _detect_faces_and_plates(self, objects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify faces and license plates from detected objects for privacy protection.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Tuple of (faces, license_plates) lists
        """
        faces = []
        license_plates = []
        
        # Handle invalid input
        if not objects or objects is None:
            return faces, license_plates
            
        # Handle tuple input like ([], None)
        if isinstance(objects, tuple):
            objects = objects[0] if objects[0] is not None else []
        
        for obj in objects:
            # Handle both dictionary and list objects
            if isinstance(obj, dict):
                label = obj.get('label', '').lower()
                description = obj.get('description', '').lower()
            else:
                # Skip non-dictionary objects
                continue
            
            # Check for faces
            if ('face' in label or 'person' in label or 'pedestrian' in label or 
                'face' in description or 'facial' in description):
                faces.append(obj)
            
            # Check for license plates
            if ('license' in label or 'plate' in label or 'number plate' in label or
                'license' in description or 'plate' in description or 'registration' in description):
                license_plates.append(obj)
        
        return faces, license_plates
    
    def apply_privacy_protection(self, image_path: str, objects: List[Dict[str, Any]], 
                               output_path: str = None, blur_faces: bool = True, 
                               blur_plates: bool = True) -> str:
        """
        Apply privacy protection by blurring faces and license plates in the image.
        
        Args:
            image_path: Path to the input image
            objects: List of detected objects
            output_path: Path to save the privacy-protected image
            blur_faces: Whether to blur detected faces
            blur_plates: Whether to blur detected license plates
            
        Returns:
            str: Path to the saved privacy-protected image
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect faces and license plates
        faces, license_plates = self._detect_faces_and_plates(objects)
        
        # Apply blurring to faces
        if blur_faces:
            for face in faces:
                bbox = self._normalize_bbox(face.get('bbox', {}))
                if bbox and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    image = self._blur_region(image, x1, y1, x2, y2, blur_strength=25)
        
        # Apply blurring to license plates
        if blur_plates:
            for plate in license_plates:
                bbox = self._normalize_bbox(plate.get('bbox', {}))
                if bbox and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    image = self._blur_region(image, x1, y1, x2, y2, blur_strength=20)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.output_dir, "privacy_protected", f"{base_name}_protected.jpg")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the privacy-protected image
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def _convert_to_label_studio_format(self, objects: List[Dict[str, Any]], 
                                       image_path: str, image_width: int, 
                                       image_height: int) -> Dict[str, Any]:
        """
        Convert detection results to Label Studio compatible JSON format.
        
        Args:
            objects: List of detected objects
            image_path: Path to the image file
            image_width, image_height: Image dimensions
            
        Returns:
            Label Studio compatible annotation dictionary
        """
        annotations = []
        
        for i, obj in enumerate(objects):
            # Skip invalid objects (empty lists, None, or non-dict objects)
            if not obj or not isinstance(obj, dict) or 'bbox' not in obj:
                continue
                
            bbox = obj['bbox']
            # Handle both list and dict bbox formats
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            elif isinstance(bbox, dict) and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            else:
                continue  # Skip objects with invalid bbox format
                
            # Get label with fallback
            label = obj.get('label', obj.get('class', 'unknown'))
            
            # Convert to Label Studio format (percentages)
            x_percent = (x1 / image_width) * 100
            y_percent = (y1 / image_height) * 100
            width_percent = ((x2 - x1) / image_width) * 100
            height_percent = ((y2 - y1) / image_height) * 100
            
            annotation = {
                "id": f"bbox_{i}",
                "type": "rectanglelabels",
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": image_width,
                "original_height": image_height
            }
            annotations.append(annotation)
        
        # Create the complete Label Studio task format
        task = {
            "data": {
                "image": image_path
            },
            "annotations": [{
                "id": 1,
                "created_username": "qwen_pipeline",
                "created_ago": "0 minutes",
                "task": 1,
                "result": annotations,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "lead_time": 0
            }],
            "predictions": [],
            "meta": {
                "model_name": self.model_name,
                "detection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_objects": len(objects)
            }
        }
        
        return task
    
    def _visualize_detections(self, image: Image.Image, objects: List[Dict[str, Any]], 
                            output_path: str) -> None:
        """
        Create visualization of detection results with bounding boxes.
        
        Args:
            image: PIL Image object
            objects: List of detected objects
            output_path: Path to save the visualization
        """
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Define colors for different object types
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, obj in enumerate(objects):
            # Skip invalid objects (empty lists, None, non-dict objects, or those without 'bbox')
            if not obj or not isinstance(obj, dict) or 'bbox' not in obj:
                continue
                
            x1, y1, x2, y2 = obj['bbox']
            color = colors[i % len(colors)]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label text with object name
            label_text = obj['label'] if obj['label'] != 'Unknown Object' else f"Object {i+1}"
            ax.text(
                x1, y1 - 5, label_text,
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        plt.title(f"Object Detection Results ({len(objects)} objects detected)", 
                 fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_objects(self, image_path: str, save_results: bool = True, apply_privacy: bool = True, use_sam_segmentation: bool = False) -> Dict[str, Any]:
        """
        Perform object detection on a single image.
        
        Args:
            image_path: Path to the input image
            save_results: Whether to save results to files
            apply_privacy: Whether to apply privacy protection (blur faces/plates)
            use_sam_segmentation: Whether to apply SAM segmentation post-processing
            
        Returns:
            Dictionary containing detection results and file paths
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
        
        # Generate detection prompt and get response
        print("Running object detection...")
        start_time = time.time()
        
        try:
            responses = self.vlm.generate([image], [self.detection_prompt])
            raw_response = responses[0] if responses else ""
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
        
        detection_time = time.time() - start_time
        print(f"Detection completed in {detection_time:.2f} seconds")
        
        # Parse the response to extract objects
        print("Parsing detection results...")
        objects = self._parse_detection_response(raw_response, image_width, image_height)
        print(f"Found {len(objects)} objects")
        
        # Apply SAM segmentation if requested and available
        segmentation_masks = []
        segmentation_visualization_path = None
        if use_sam_segmentation and self.enable_sam and objects:
            try:
                print("Applying SAM segmentation...")
                segmentation_start_time = time.time()
                segmentation_masks, segmentation_visualization_path = self._apply_sam_segmentation(
                    image, objects, image_path, "vlm"
                )
                segmentation_time = time.time() - segmentation_start_time
                print(f"SAM segmentation completed in {segmentation_time:.2f} seconds")
            except Exception as e:
                print(f"Warning: SAM segmentation failed: {str(e)}")
        
        # Apply privacy protection if requested
        privacy_protected_path = None
        if apply_privacy and objects:
            try:
                privacy_protected_path = self.apply_privacy_protection(image_path, objects)
                print(f"Privacy protection applied, saved to: {privacy_protected_path}")
            except Exception as e:
                print(f"Warning: Privacy protection failed: {str(e)}")
        
        # Prepare results
        results = {
            'image_path': image_path,
            'image_width': image_width,
            'image_height': image_height,
            'objects': objects,
            'raw_response': raw_response,
            'detection_time': detection_time,
            'model_name': self.model_name,
            'privacy_protected_path': privacy_protected_path,
            'segmentation_masks': segmentation_masks,
            'segmentation_visualization_path': segmentation_visualization_path
        }
        
        if save_results:
            # Generate output file names
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save raw response
            raw_file = os.path.join(self.raw_dir, f"{base_name}_{timestamp}_raw.txt")
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Detection Time: {detection_time:.2f}s\n")
                f.write(f"Objects Found: {len(objects)}\n")
                f.write("\n" + "="*50 + "\n")
                f.write(raw_response)
            
            # Convert to Label Studio format and save
            label_studio_data = self._convert_to_label_studio_format(
                objects, image_path, image_width, image_height
            )
            
            json_file = os.path.join(self.json_dir, f"{base_name}_{timestamp}_annotations.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
            
            # Create visualization
            viz_file = os.path.join(self.viz_dir, f"{base_name}_{timestamp}_detection.png")
            self._visualize_detections(image, objects, viz_file)
            
            # Add file paths to results
            results.update({
                'raw_response_file': raw_file,
                'json_annotation_file': json_file,
                'visualization_file': viz_file
            })
            
            print(f"Results saved:")
            print(f"  - Raw response: {raw_file}")
            print(f"  - JSON annotations: {json_file}")
            print(f"  - Visualization: {viz_file}")
            if segmentation_visualization_path:
                print(f"  - SAM Segmentation: {segmentation_visualization_path}")
        
        return results

    def detect_objects_hybrid(self, image_path: str, save_results: bool = True, apply_privacy: bool = True, 
                            use_sam_segmentation: bool = False, ensemble_method: str = 'nms', 
                            box_merge_threshold: float = 0.3, sequential_mode: bool = False) -> Dict[str, Any]:
        """
        Perform hybrid object detection combining traditional detectors with Qwen2.5-VL.
        
        This method supports two modes:
        
        Standard mode (sequential_mode=False):
        1. Runs traditional detectors and VLM in parallel
        2. Ensembles all results together
        
        Sequential mode (sequential_mode=True):
        1. First runs traditional object detectors to get bounding boxes
        2. Uses VLM to analyze only the detected bounding boxes
        3. Replaces traditional detector results with VLM analysis of those boxes
        
        Args:
            image_path: Path to the input image
            save_results: Whether to save results to files
            apply_privacy: Whether to apply privacy protection (blur faces/plates)
            use_sam_segmentation: Whether to apply SAM segmentation post-processing
            ensemble_method: Method for ensembling traditional detectors ('nms' or 'wbf')
            box_merge_threshold: Threshold for merging nearby/overlapping boxes
            sequential_mode: If True, use sequential detection (traditional first, then VLM on boxes)
            
        Returns:
            Dictionary containing detection results and file paths
        """
        print(f"Processing image with hybrid detection: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
        
        start_time = time.time()
        all_objects = []
        traditional_detections = []
        
        # Step 1: Run traditional detectors if enabled
        if self.enable_traditional_detectors and self.traditional_detectors:
            print("Running traditional object detectors...")
            
            for detector in self.traditional_detectors:
                try:
                    detections = detector.detect(image)
                    # Map COCO classes to allowed classes for traditional detectors
                    mapped_detections = []
                    for det in detections:
                        mapped_label = map_coco_to_allowed_classes(det['label'])
                        mapped_det = det.copy()
                        mapped_det['label'] = mapped_label
                        mapped_det['original_coco_label'] = det['label']  # Keep original for reference
                        mapped_detections.append(mapped_det)
                    
                    traditional_detections.append(mapped_detections)
                    print(f"  {detector.__class__.__name__}: {len(detections)} detections")
                except Exception as e:
                    print(f"  Warning: {detector.__class__.__name__} failed: {str(e)}")
                    continue
            
        # Step 2: Run Qwen2.5-VL analysis
        vlm_detections = []
        
        if sequential_mode and traditional_detections:
            # Sequential mode: analyze specific bounding boxes from traditional detectors
            print("Running sequential VLM analysis on traditional detector bounding boxes...")
            
            # Flatten all traditional detections
            all_traditional_boxes = []
            for detector_detections in traditional_detections:
                all_traditional_boxes.extend(detector_detections)
            
            if all_traditional_boxes:
                # Extract bounding box tuples for prompt generation
                bbox_tuples = [tuple(det['bbox']) for det in all_traditional_boxes]
                # Generate bbox-specific prompt
                bbox_prompt = self._generate_bbox_specific_prompt(bbox_tuples)
                
                try:
                    responses = self.vlm.generate([image], [bbox_prompt])
                    raw_response = responses[0] if responses else ""
                    
                    # Parse bbox-specific VLM response
                    vlm_objects = self._parse_bbox_specific_response(raw_response, all_traditional_boxes)
                    
                    # Convert VLM objects to detection format for ensemble
                    for obj in vlm_objects:
                        vlm_detections.append({
                            'label': obj['label'],
                            'bbox': obj['bbox'],
                            'confidence': obj.get('confidence', 0.8),
                            'description': obj.get('description', f"VLM analysis: {obj['label']}"),
                            'source': 'vlm_sequential'
                        })
                        
                except Exception as e:
                    print(f"Warning: Sequential VLM detection failed: {str(e)}")
            else:
                print("No traditional detections found for sequential analysis")
        else:
            # Standard mode: comprehensive VLM analysis
            print("Running Qwen2.5-VL for comprehensive analysis...")
            try:
                responses = self.vlm.generate([image], [self.detection_prompt])
                raw_response = responses[0] if responses else ""
                
                # Parse VLM response
                vlm_objects = self._parse_detection_response(raw_response, image_width, image_height)
                
                # Convert VLM objects to detection format for ensemble
                for obj in vlm_objects:
                    vlm_detections.append({
                        'label': obj['label'],
                        'bbox': obj['bbox'],
                        'confidence': obj.get('confidence', 0.8),  # Default confidence for VLM
                        'description': obj.get('description', f"Detected by VLM: {obj['label']}"),
                        'source': 'vlm'
                    })
                    
            except Exception as e:
                print(f"Warning: VLM detection failed: {str(e)}")
        
        # Step 3: Ensemble all detections
        if sequential_mode and vlm_detections:
            # In sequential mode, VLM results take priority as they refine traditional detector boxes
            print("Using sequential ensemble: VLM results take priority")
            all_objects = vlm_detections.copy()
            
            # Add any traditional detections that weren't analyzed by VLM
            analyzed_boxes = {tuple(obj['bbox']) for obj in vlm_detections}
            for detector_detections in traditional_detections:
                for det in detector_detections:
                    if tuple(det['bbox']) not in analyzed_boxes:
                        all_objects.append({
                            'label': det['label'],
                            'bbox': det['bbox'],
                            'confidence': det['confidence'],
                            'description': f"Traditional detector: {det['label']}",
                            'source': det.get('source', 'traditional')
                        })
        else:
            # Standard ensemble using hybrid ensemble function
            all_objects = ensemble_hybrid_vlm_detections(
                traditional_detections=traditional_detections,
                vlm_detections=vlm_detections,
                ensemble_method=ensemble_method,
                box_merge_threshold=box_merge_threshold,
                matching_method='overlap',  # Use overlap method for better matching
                iou_threshold=0.3,
                overlap_threshold=0.5
            )
        
        detection_time = time.time() - start_time
        print(f"Hybrid detection completed in {detection_time:.2f} seconds")
        print(f"Total objects detected: {len(all_objects)}")
        
        # Apply SAM segmentation if requested
        # Initialize segmentation visualization path
        segmentation_visualization_path = None
        
        if use_sam_segmentation and self.enable_sam and all_objects:
            print("Applying SAM segmentation...")
            seg_dir = os.path.join(self.output_dir, "segmentations")
            os.makedirs(seg_dir, exist_ok=True)
            
            try:
                segmentation_masks, segmentation_visualization_path = self._apply_sam_segmentation(image, all_objects, image_path, "hybrid")
                print(f"SAM segmentation completed with {len(segmentation_masks)} masks")
            except Exception as e:
                print(f"Warning: SAM segmentation failed: {str(e)}")
        
        # Apply privacy protection if requested
        if apply_privacy:
            faces, license_plates = self._detect_faces_and_plates(all_objects)
            if faces or license_plates:
                print(f"Applying privacy protection: {len(faces)} faces, {len(license_plates)} license plates")
                privacy_objects = faces + license_plates
                privacy_protected_path = self.apply_privacy_protection(image_path, privacy_objects)
        
        # Prepare result
        result = {
            'image_path': image_path,
            'objects': all_objects,
            'detection_time': detection_time,
            'image_dimensions': {'width': image_width, 'height': image_height},
            'detection_method': 'hybrid',
            'traditional_detectors_used': len(self.traditional_detectors) if self.traditional_detectors else 0,
            'vlm_used': True,
            'sam_segmentation': use_sam_segmentation and self.enable_sam
        }
        
        if save_results:
            # Generate output file names
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save raw response (create a synthetic raw response for hybrid method)
            raw_response = f"Hybrid Detection Results\nTraditional Detectors: {len(self.traditional_detectors) if self.traditional_detectors else 0}\nVLM Used: True\nTotal Objects: {len(all_objects)}\nDetection Time: {detection_time:.2f}s\n\nDetected Objects:\n"
            for i, obj in enumerate(all_objects):
                raw_response += f"{i+1}. {obj['label']} - {obj['description']}\n"
            
            raw_file = os.path.join(self.raw_dir, f"{base_name}_{timestamp}_hybrid_raw.txt")
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Detection Time: {detection_time:.2f}s\n")
                f.write(f"Objects Found: {len(all_objects)}\n")
                f.write(f"Detection Method: hybrid\n")
                f.write("\n" + "="*50 + "\n")
                f.write(raw_response)
            
            # Convert to Label Studio format and save
            label_studio_data = self._convert_to_label_studio_format(
                all_objects, image_path, image_width, image_height
            )
            
            json_file = os.path.join(self.json_dir, f"{base_name}_{timestamp}_hybrid_annotations.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
            
            # Create visualization
            viz_file = os.path.join(self.viz_dir, f"{base_name}_{timestamp}_hybrid_detection.png")
            self._visualize_detections(image, all_objects, viz_file)
            
            # Save processed image (with privacy protection if applied)
            if apply_privacy:
                processed_filename = f"{base_name}_{timestamp}_hybrid_processed.png"
                processed_path = os.path.join(self.output_dir, processed_filename)
                image.save(processed_path)
                result['processed_image_path'] = processed_path
            
            # Add file paths to results
            result.update({
                'raw_response_file': raw_file,
                'json_annotation_file': json_file,
                'visualization_file': viz_file
            })
            
            print(f"Results saved:")
            print(f"  - Raw response: {raw_file}")
            print(f"  - JSON annotations: {json_file}")
            print(f"  - Visualization: {viz_file}")
            if segmentation_visualization_path:
                print(f"  - SAM Segmentation: {segmentation_visualization_path}")
        
        return result

    def detect_objects_batch(self, image_paths: List[str], save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Perform object detection on multiple images.
        
        Args:
            image_paths: List of paths to input images
            save_results: Whether to save results to files
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                result = self.detect_objects(image_path, save_results)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'objects': []
                })
        
        if save_results:
            # Save batch summary
            summary_file = os.path.join(self.output_dir, f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
            summary = {
                'total_images': len(image_paths),
                'successful_detections': len([r for r in results if 'error' not in r]),
                'failed_detections': len([r for r in results if 'error' in r]),
                'total_objects_detected': sum(len(r.get('objects', [])) for r in results),
                'model_name': self.model_name,
                'processing_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': results
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\nBatch summary saved: {summary_file}")
        
        return results


    def extract_video_metadata(self, video_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Extract metadata including GPS information from video file using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (metadata, gps_info, creation_time)
        """
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            '-show_streams', 
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            metadata = json.loads(result.stdout)
            
            # Extract GPS data if available
            gps_info = {}
            creation_time = None
            
            # Look for creation time and GPS data in format tags
            if 'format' in metadata and 'tags' in metadata['format']:
                tags = metadata['format']['tags']
                
                # Extract creation time
                time_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate']
                for field in time_fields:
                    if field in tags:
                        creation_time = tags[field]
                        break
                
                # Common GPS metadata fields
                gps_fields = [
                    'location', 'location-eng', 'GPS', 
                    'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
                    'com.apple.quicktime.location.ISO6709'
                ]
                
                for field in gps_fields:
                    if field in tags:
                        gps_info[field] = tags[field]
            
            # Also check stream metadata for creation time if not found
            if creation_time is None and 'streams' in metadata:
                for stream in metadata['streams']:
                    if 'tags' in stream and 'creation_time' in stream['tags']:
                        creation_time = stream['tags']['creation_time']
                        break
            
            # If no creation time found, use file modification time
            if creation_time is None:
                file_mtime = os.path.getmtime(video_path)
                creation_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
            
            return metadata, gps_info, creation_time
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}, {}, None
    
    def _resize_with_aspect_ratio(self, image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image maintaining aspect ratio.
        
        Args:
            image: PIL Image or numpy array
            target_size: Tuple of (width, height) representing the maximum dimensions
            
        Returns:
            Resized PIL Image
        """
        if isinstance(image, np.ndarray):
            # Convert OpenCV image (numpy array) to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Handle both integer and tuple inputs for target_size
        if isinstance(target_size, (int, float)):
            # If target_size is a single number, use it as the maximum dimension
            if original_width > original_height:
                target_width = int(target_size)
                target_height = int(target_size * original_height / original_width)
            else:
                target_height = int(target_size)
                target_width = int(target_size * original_width / original_height)
        else:
            # If target_size is a tuple, use it directly
            target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        # Determine new dimensions maintaining aspect ratio
        if original_aspect > target_aspect:
            # Width constrained
            new_width = target_width
            new_height = int(target_width / original_aspect)
        else:
            # Height constrained
            new_height = target_height
            new_width = int(target_height * original_aspect)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_image
    
    def process_video_frames(self, video_path: str, target_size: Tuple[int, int] = (640, 480), 
                           extraction_method: str = "scene_change", scenechange_threshold: float = 10.0,
                           save_results: bool = True, use_sam_segmentation: bool = False, 
                           batch_size: int = 4) -> Dict[str, Any]:
        """
        Extract key frames from a video file and perform object detection on each frame.
        
        Args:
            video_path: Path to the input video file
            target_size: Tuple of (width, height) maximum dimensions for resizing
            extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
            scenechange_threshold: Threshold for scene change detection
            save_results: Whether to save results to files
            use_sam_segmentation: Whether to apply SAM segmentation to detected objects
            batch_size: Number of frames to process in batch for better GPU utilization
            
        Returns:
            Dictionary containing video processing results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ignore video files that are less than 2 seconds long
        if duration < 2.0:
            cap.release()
            raise ValueError(f"Video too short (duration: {duration:.2f} seconds, minimum required: 2.0 seconds)")
        
        # Get video metadata
        metadata, gps_info, creation_time = self.extract_video_metadata(video_path)
        
        print(f"Video Information:")
        print(f"- Frame Rate: {fps} fps")
        print(f"- Frame Count: {frame_count}")
        print(f"- Resolution: {original_width}x{original_height}")
        print(f"- Duration: {duration:.2f} seconds")
        print(f"- Creation Time: {creation_time}")
        
        # Create video-specific output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, f"video_{video_name}_{self._get_timestamp()}")
        
        if save_results:
            os.makedirs(video_output_dir, exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "frames"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "raw_responses"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "json_annotations"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "visualizations"), exist_ok=True)
            if use_sam_segmentation and self.enable_sam:
                os.makedirs(os.path.join(video_output_dir, "segmentations"), exist_ok=True)
        
        # Initialize variables for frame extraction
        prev_frame = None
        frame_idx = 0
        saved_count = 0
        extracted_frames = []
        all_results = []
        
        # Parameters for scene change detection
        min_scene_change_threshold = scenechange_threshold
        frame_interval = int(fps) * 1  # Save a frame every second as fallback
        
        print(f"Extracting frames using method: {extraction_method}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for scene change detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            should_save = False
            reason = ""
            
            # Method 1: Detect scene changes
            if extraction_method in ["scene_change", "both"]:
                if prev_frame is not None:
                    # Calculate absolute difference between frames
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > min_scene_change_threshold:
                        should_save = True
                        reason = f"scene_change (diff: {mean_diff:.2f})"
            
            # Method 2: Extract frames at regular intervals
            if extraction_method in ["interval", "both"]:
                if frame_idx % frame_interval == 0:
                    should_save = True
                    reason = f"interval ({frame_idx // frame_interval}s)"
            
            # Save frame if criteria met
            if should_save:
                # Calculate timestamp
                timestamp_seconds = frame_idx / fps
                timestamp = f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}.{int((timestamp_seconds % 1) * 1000):03d}"
                
                # Resize frame
                resized_frame_pil = self._resize_with_aspect_ratio(frame, target_size)
                new_width, new_height = resized_frame_pil.size
                
                # Save frame if requested
                frame_filename = None
                if save_results:
                    frame_filename = f"{video_name}_frame_{saved_count:04d}_{timestamp.replace(':', '-').replace('.', '_')}.jpg"
                    frame_path = os.path.join(video_output_dir, "frames", frame_filename)
                    resized_frame_pil.save(frame_path, "JPEG", quality=95)
                
                # Add frame to batch for processing
                frame_data = {
                    'image': resized_frame_pil,
                    'saved_index': saved_count,
                    'timestamp_seconds': timestamp_seconds,
                    'timestamp': timestamp,
                    'extraction_reason': reason,
                    'original_dimensions': {'width': original_width, 'height': original_height},
                    'resized_dimensions': {'width': new_width, 'height': new_height},
                    'frame_filename': frame_filename,
                    'frame_index': frame_idx
                }
                
                # Store frame data for batch processing
                if not hasattr(self, '_frame_batch'):
                    self._frame_batch = []
                
                self._frame_batch.append(frame_data)
                saved_count += 1
                
                print(f"Added frame {saved_count} to batch: {timestamp} - {reason}")
                
                # Process batch when it reaches the specified size
                if len(self._frame_batch) >= batch_size:
                    try:
                        batch_results = self._process_frame_batch(
                            self._frame_batch, video_output_dir, video_name, 
                            use_sam_segmentation, save_results=save_results
                        )
                        
                        # Add results and frame info
                        for i, (frame_data, frame_results) in enumerate(zip(self._frame_batch, batch_results)):
                            frame_info = {
                                'frame_index': frame_data['frame_index'],
                                'saved_index': frame_data['saved_index'],
                                'timestamp_seconds': frame_data['timestamp_seconds'],
                                'timestamp': frame_data['timestamp'],
                                'extraction_reason': frame_data['extraction_reason'],
                                'original_dimensions': frame_data['original_dimensions'],
                                'resized_dimensions': frame_data['resized_dimensions'],
                                'frame_filename': frame_data['frame_filename'],
                                'detection_results': frame_results
                            }
                            
                            extracted_frames.append(frame_info)
                            all_results.append(frame_results)
                            
                            print(f"Processed frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(frame_results.get('objects', []))} objects detected")
                        
                        # Clear batch
                        self._frame_batch = []
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        # Add empty results for failed batch
                        for frame_data in self._frame_batch:
                            frame_info = {
                                'frame_index': frame_data['frame_index'],
                                'saved_index': frame_data['saved_index'],
                                'timestamp_seconds': frame_data['timestamp_seconds'],
                                'timestamp': frame_data['timestamp'],
                                'extraction_reason': frame_data['extraction_reason'],
                                'original_dimensions': frame_data['original_dimensions'],
                                'resized_dimensions': frame_data['resized_dimensions'],
                                'frame_filename': frame_data['frame_filename'],
                                'detection_results': {
                                    'objects': [],
                                    'raw_response': '',
                                    'inference_time': 0,
                                    'model_name': self.model_name,
                                    'segmentation_masks': [],
                                    'segmentation_visualization_path': None,
                                    'error': str(e)
                                }
                            }
                            extracted_frames.append(frame_info)
                            all_results.append(frame_info['detection_results'])
                        
                        self._frame_batch = []
            
            prev_frame = gray.copy()
            frame_idx += 1
        
        cap.release()
        
        # Process any remaining frames in the batch
        if hasattr(self, '_frame_batch') and self._frame_batch:
            try:
                print(f"Processing final batch of {len(self._frame_batch)} frames...")
                batch_results = self._process_frame_batch(
                    self._frame_batch, video_output_dir, video_name, 
                    use_sam_segmentation, save_results=save_results
                )
                
                # Add results and frame info
                for i, (frame_data, frame_results) in enumerate(zip(self._frame_batch, batch_results)):
                    frame_info = {
                        'frame_index': frame_data['frame_index'],
                        'saved_index': frame_data['saved_index'],
                        'timestamp_seconds': frame_data['timestamp_seconds'],
                        'timestamp': frame_data['timestamp'],
                        'extraction_reason': frame_data['extraction_reason'],
                        'original_dimensions': frame_data['original_dimensions'],
                        'resized_dimensions': frame_data['resized_dimensions'],
                        'frame_filename': frame_data['frame_filename'],
                        'detection_results': frame_results
                    }
                    
                    extracted_frames.append(frame_info)
                    all_results.append(frame_results)
                    
                    print(f"Processed final frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(frame_results.get('objects', []))} objects detected")
                
                # Clear batch
                self._frame_batch = []
                
            except Exception as e:
                print(f"Error processing final batch: {e}")
                # Add empty results for failed batch
                for frame_data in self._frame_batch:
                    frame_info = {
                        'frame_index': frame_data['frame_index'],
                        'saved_index': frame_data['saved_index'],
                        'timestamp_seconds': frame_data['timestamp_seconds'],
                        'timestamp': frame_data['timestamp'],
                        'extraction_reason': frame_data['extraction_reason'],
                        'original_dimensions': frame_data['original_dimensions'],
                        'resized_dimensions': frame_data['resized_dimensions'],
                        'frame_filename': frame_data['frame_filename'],
                        'detection_results': {
                            'objects': [],
                            'raw_response': '',
                            'inference_time': 0,
                            'model_name': self.model_name,
                            'segmentation_masks': [],
                            'segmentation_visualization_path': None,
                            'error': str(e)
                        }
                    }
                    extracted_frames.append(frame_info)
                    all_results.append(frame_info['detection_results'])
                
                self._frame_batch = []
        
        # Create summary results
        total_objects = sum(len(result.get('objects', [])) for result in all_results)
        
        video_results = {
            'video_path': video_path,
            'video_metadata': {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'original_width': original_width,
                'original_height': original_height,
                'creation_time': creation_time,
                'gps_info': gps_info
            },
            'extraction_settings': {
                'method': extraction_method,
                'scene_change_threshold': scenechange_threshold,
                'target_size': target_size
            },
            'extracted_frames': extracted_frames,
            'summary': {
                'total_frames_processed': frame_idx,
                'frames_extracted': saved_count,
                'total_objects_detected': total_objects,
                'processing_time': time.time()
            }
        }
        
        # Save video summary if requested
        if save_results:
            summary_path = os.path.join(video_output_dir, f"{video_name}_video_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(video_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\nVideo processing complete!")
            print(f"- Processed {frame_idx} total frames")
            print(f"- Extracted {saved_count} key frames")
            print(f"- Detected {total_objects} total objects")
            print(f"- Results saved to: {video_output_dir}")
        
        return video_results
    
    def _process_frame_batch(self, frame_batch: List[Dict[str, Any]], video_output_dir: str, 
                           video_name: str, use_sam_segmentation: bool = False, 
                           save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of frames efficiently with GPU memory optimization.
        
        Args:
            frame_batch: List of frame dictionaries with image data and metadata
            video_output_dir: Output directory for saving results
            video_name: Base name of the video
            use_sam_segmentation: Whether to apply SAM segmentation
            save_results: Whether to save results to files
            
        Returns:
            List of processing results for each frame
        """
        if not frame_batch:
            return []
        
        batch_results = []
        
        try:
            # Extract images from batch
            images = [frame_data['image'] for frame_data in frame_batch]
            prompts = [self.detection_prompt] * len(images)
            
            # Batch inference for better GPU utilization
            print(f"Processing batch of {len(images)} frames...")
            start_time = time.time()
            
            # Process all images in batch
            responses = self.vlm.generate(images, prompts)
            batch_inference_time = time.time() - start_time
            print(f"Batch inference completed in {batch_inference_time:.2f} seconds")
            
            # Process each frame result
            for i, (frame_data, response) in enumerate(zip(frame_batch, responses)):
                try:
                    image = frame_data['image']
                    image_width, image_height = image.size
                    
                    # Parse detection response
                    objects = self._parse_detection_response(response, image_width, image_height)
                    
                    # Apply SAM segmentation if requested
                    segmentation_masks = []
                    segmentation_visualization_path = None
                    if use_sam_segmentation and self.enable_sam and objects:
                        try:
                            segmentation_masks, segmentation_visualization_path = self._apply_sam_segmentation(
                                image, objects, f"frame_{frame_data['saved_index']:04d}", "vlm"
                            )
                            
                            # Save segmentation to video directory
                            if segmentation_visualization_path and save_results:
                                seg_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_segmentation.png"
                                final_seg_path = os.path.join(video_output_dir, "segmentations", seg_filename)
                                
                                # Copy segmentation result to video directory
                                import shutil
                                shutil.copy2(segmentation_visualization_path, final_seg_path)
                                segmentation_visualization_path = final_seg_path
                                
                        except Exception as e:
                            print(f"Warning: SAM segmentation failed for frame {frame_data['saved_index']}: {str(e)}")
                    
                    # Create frame results
                    frame_results = {
                        'objects': objects,
                        'raw_response': response,
                        'inference_time': batch_inference_time / len(images),  # Approximate per-frame time
                        'model_name': self.model_name,
                        'segmentation_masks': segmentation_masks,
                        'segmentation_visualization_path': segmentation_visualization_path
                    }
                    
                    # Save individual frame results if requested
                    if save_results:
                        # Save raw response
                        raw_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_raw.txt"
                        raw_path = os.path.join(video_output_dir, "raw_responses", raw_filename)
                        with open(raw_path, 'w', encoding='utf-8') as f:
                            f.write(response)
                        
                        # Save JSON annotations
                        json_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_annotations.json"
                        json_path = os.path.join(video_output_dir, "json_annotations", json_filename)
                        
                        # Create Label Studio format for this frame
                        frame_annotation = self._convert_to_label_studio_format(
                            objects, 
                            frame_data.get('frame_filename', f"frame_{frame_data['saved_index']:04d}.jpg"), 
                            image_width, 
                            image_height
                        )
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(frame_annotation, f, indent=2, ensure_ascii=False)
                        
                        # Save visualization
                        if objects:
                            vis_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_detection.png"
                            vis_path = os.path.join(video_output_dir, "visualizations", vis_filename)
                            self._visualize_detections(image, objects, vis_path)
                    
                    batch_results.append(frame_results)
                    
                    print(f"Processed frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(objects)} objects detected")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_data['saved_index']}: {e}")
                    # Add empty result to maintain batch consistency
                    batch_results.append({
                        'objects': [],
                        'raw_response': '',
                        'inference_time': 0,
                        'model_name': self.model_name,
                        'segmentation_masks': [],
                        'segmentation_visualization_path': None,
                        'error': str(e)
                    })
            
            # Clear GPU cache after batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Return empty results for the batch
            batch_results = [{
                'objects': [],
                'raw_response': '',
                'inference_time': 0,
                'model_name': self.model_name,
                'segmentation_masks': [],
                'segmentation_visualization_path': None,
                'error': str(e)
            } for _ in frame_batch]
        
        return batch_results
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2] format
            bbox2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_sam_segmentation(self, image: Image.Image, objects: List[Dict[str, Any]], 
                               image_path: str, detection_type: str = "vlm") -> Tuple[List[Dict[str, Any]], str]:
        """
        Apply SAM segmentation to detected objects within their bounding box regions only.
        Updates bounding boxes based on segmentation results for better alignment.
        
        Args:
            image: PIL Image object
            objects: List of detected objects with bounding boxes
            image_path: Path to the original image
            detection_type: Type of detection ('vlm' or 'hybrid') for filename
            
        Returns:
            Tuple of (segmentation_masks, visualization_path)
        """
        if not self.enable_sam or not objects:
            return [], None
        
        try:
            # Convert PIL image to numpy array for processing
            image_np = np.array(image)
            updated_objects = []
            segmentation_masks = []
            mask_overlays = []
            
            # Process each object individually to constrain segmentation to bounding box
            for i, obj in enumerate(objects):
                # Skip invalid objects (empty lists, None, non-dict objects, or those without 'bbox')
                if not obj or not isinstance(obj, dict) or 'bbox' not in obj:
                    continue
                    
                bbox = obj['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Extract the bounding box region from the image
                bbox_region = image.crop((x1, y1, x2, y2))
                
                # Use center point of the cropped region as prompt
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x_local = bbox_width // 2
                center_y_local = bbox_height // 2
                
                # Process the cropped region with SAM
                inputs = self.sam_processor(
                    bbox_region, 
                    input_points=[[[center_x_local, center_y_local]]], 
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)
                
                # Get masks for this specific region
                masks = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), 
                    inputs["original_sizes"].cpu(), 
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                if masks and len(masks[0]) > 0:
                    # Get the first mask (best segmentation)
                    mask = masks[0][0]
                    
                    # Convert mask to numpy
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.squeeze().cpu().numpy()
                    else:
                        mask_np = np.array(mask).squeeze()
                    
                    # Ensure mask is 2D
                    if mask_np.ndim > 2:
                        mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.max(axis=0)
                    
                    # Convert to binary mask
                    binary_mask = (mask_np > 0.5).astype(np.uint8)
                    
                    # Find the tight bounding box of the segmented region
                    mask_coords = np.where(binary_mask > 0)
                    if len(mask_coords[0]) > 0:  # If mask contains any pixels
                        # Get local coordinates of the segmented region
                        local_y_min, local_y_max = mask_coords[0].min(), mask_coords[0].max()
                        local_x_min, local_x_max = mask_coords[1].min(), mask_coords[1].max()
                        
                        # Convert back to global coordinates
                        new_x1 = int(x1 + local_x_min)
                        new_y1 = int(y1 + local_y_min)
                        new_x2 = int(x1 + local_x_max + 1)  # +1 for inclusive bounds
                        new_y2 = int(y1 + local_y_max + 1)
                        
                        # Update the object with refined bounding box
                        updated_obj = obj.copy()
                        updated_obj['bbox'] = [new_x1, new_y1, new_x2, new_y2]
                        updated_obj['original_bbox'] = bbox  # Keep original for reference
                        updated_objects.append(updated_obj)
                        
                        # Create full-size mask for visualization
                        full_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
                        full_mask[y1:y2, x1:x2] = binary_mask * 255
                        
                        # Store mask info with updated bbox
                        mask_info = {
                            'object_id': i,
                            'object_label': obj['label'],
                            'mask_shape': full_mask.shape,
                            'bbox': [new_x1, new_y1, new_x2, new_y2],
                            'original_bbox': bbox,
                            'mask_array': full_mask
                        }
                        segmentation_masks.append(mask_info)
                        
                        # Create colored overlay for visualization
                        color = plt.cm.tab10(i % 10)[:3]  # Get color from colormap
                        colored_mask = np.zeros((*full_mask.shape, 3), dtype=np.uint8)
                        mask_indices = full_mask > 0
                        colored_mask[mask_indices] = [int(c * 255) for c in color]
                        mask_overlays.append(colored_mask)
                    else:
                        # If no valid mask, keep original object
                        updated_objects.append(obj)
                else:
                    # If SAM failed for this object, keep original
                    updated_objects.append(obj)
            
            # Create visualization with segmentation, bounding boxes, and labels
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = self._get_timestamp()
            if detection_type == "hybrid":
                viz_path = os.path.join(self.seg_dir, f"{base_name}_{timestamp}_hybrid_segmentation.png")
            else:
                viz_path = os.path.join(self.seg_dir, f"{base_name}_{timestamp}_segmentation.png")
            
            # Combine original image with mask overlays
            combined_image = image_np.copy()
            for mask_overlay in mask_overlays:
                # Alpha blend the mask with the original image
                alpha = 0.5
                mask_indices = np.any(mask_overlay > 0, axis=2)
                combined_image[mask_indices] = (
                    alpha * mask_overlay[mask_indices] + 
                    (1 - alpha) * combined_image[mask_indices]
                ).astype(np.uint8)
            
            # Convert to PIL for drawing bounding boxes and labels
            combined_pil = Image.fromarray(combined_image)
            draw = ImageDraw.Draw(combined_pil)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Define colors for bounding boxes (matching segmentation colors)
            bbox_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            
            # Draw both original and updated bounding boxes for comparison
            for i, obj in enumerate(updated_objects):
                # Skip invalid objects (empty lists, None, non-dict objects, or those without 'bbox')
                if not obj or not isinstance(obj, dict) or 'bbox' not in obj:
                    continue
                    
                if i < len(segmentation_masks):
                    # Draw original bounding box in lighter color
                    if 'original_bbox' in obj:
                        orig_bbox = obj['original_bbox']
                        orig_x1, orig_y1, orig_x2, orig_y2 = orig_bbox
                        color = bbox_colors[i % len(bbox_colors)]
                        draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline=color, width=1)
                    
                    # Draw updated bounding box in bold
                    bbox = obj['bbox']
                    x1, y1, x2, y2 = bbox
                    color = bbox_colors[i % len(bbox_colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Prepare label text
                    label_text = obj['label'] if obj['label'] != 'Unknown Object' else f"Object {i+1}"
                    
                    # Calculate text size and position
                    if font:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        # Estimate text size if font is not available
                        text_width = len(label_text) * 8
                        text_height = 12
                    
                    # Position label above bounding box
                    text_x = x1
                    text_y = max(0, y1 - text_height - 5)
                    
                    # Draw background rectangle for text
                    draw.rectangle(
                        [text_x - 2, text_y - 2, text_x + text_width + 4, text_y + text_height + 2],
                        fill='white', outline=color, width=2
                    )
                    
                    # Draw text
                    if font:
                        draw.text((text_x, text_y), label_text, fill=color, font=font)
                    else:
                        draw.text((text_x, text_y), label_text, fill=color)
            
            # Save enhanced visualization
            combined_pil.save(viz_path)
            
            # Update the original objects list with refined bounding boxes
            for i, updated_obj in enumerate(updated_objects):
                if i < len(objects):
                    objects[i] = updated_obj
            
            return segmentation_masks, viz_path
            
        except Exception as e:
            print(f"Error in SAM segmentation: {e}")
            return [], None
    
    def detect_objects_from_image(self, image: Image.Image, save_results: bool = True, use_sam_segmentation: bool = False) -> Dict[str, Any]:
        """
        Perform object detection on a PIL Image object.
        
        Args:
            image: PIL Image object
            save_results: Whether to save results to files
            use_sam_segmentation: Whether to apply SAM segmentation to detected objects
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Get image dimensions
            image_width, image_height = image.size
            
            # Generate detection prompt
            prompt = self.detection_prompt
            
            # Perform inference
            start_time = time.time()
            response = self.vlm.generate([image], [prompt])[0]
            inference_time = time.time() - start_time
            
            # Parse the response
            objects = self._parse_detection_response(response, image_width, image_height)
            
            # Apply SAM segmentation if requested and enabled
            if use_sam_segmentation and self.enable_sam and objects:
                # Create segmentations directory
                seg_dir = os.path.join(self.output_dir, "segmentations")
                os.makedirs(seg_dir, exist_ok=True)
                
                # Apply SAM segmentation to detected objects
                segmentation_masks, segmentation_viz_path = self._apply_sam_segmentation(image, objects, "image_from_pil", "vlm")
            
            return {
                'objects': objects,
                'raw_response': response,
                'inference_time': inference_time,
                'image_dimensions': {'width': image_width, 'height': image_height}
            }
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return {
                'objects': [],
                'raw_response': '',
                'inference_time': 0,
                'image_dimensions': {'width': 0, 'height': 0},
                'error': str(e)
            }

def main():
    """
    Example usage of the QwenObjectDetectionPipeline.
    """
    # Initialize the pipeline with traditional detectors and SAM segmentation enabled
    pipeline = QwenObjectDetectionPipeline(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        output_dir="./output/qwen_detection_results",
        enable_sam=True,  # Enable SAM segmentation capabilities
        enable_traditional_detectors=True,  # Enable traditional object detectors
        traditional_detectors=['yolo', 'detr']  # Use YOLO and DETR detectors
    )
    
    # Example: Process a single image with hybrid detection
    image_path = "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/sj2.jpg"
    
    # Traditional Qwen2.5-VL only detection
    results_vlm = pipeline.detect_objects(image_path, use_sam_segmentation=True)
    print(f"VLM-only detection: {len(results_vlm['objects'])} objects")
    
    # Hybrid detection (traditional + VLM)
    results_hybrid = pipeline.detect_objects_hybrid(image_path, use_sam_segmentation=True, sequential_mode=True, save_results=True)
    print(f"Hybrid detection: {len(results_hybrid['objects'])} objects")
    print(f"Traditional detectors used: {results_hybrid['traditional_detectors_used']}")
    
    # Display detected objects for hybrid detection
    if results_hybrid['objects']:
        print("\nHybrid Detection Results:")
        for i, obj in enumerate(results_hybrid['objects']):
            # Handle the actual object structure returned by the pipeline
            if isinstance(obj, dict):
                # Extract information from the actual object structure
                bbox = obj.get('bbox', [])
                label = obj.get('object_label', obj.get('label', 'Unknown'))
                object_id = obj.get('object_id', i+1)
                source = obj.get('source', 'SAM/Traditional')
                
                print(f"  Object {i+1}: {label} (ID: {object_id}, source: {source})")
                if bbox and len(bbox) >= 4:
                    # Handle both list and nested dict bbox formats
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        print(f"    Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    
                # Show mask information if available
                if 'mask_shape' in obj:
                    print(f"    Mask shape: {obj['mask_shape']}")
            else:
                print(f"  Object {i+1}: Unknown format - {type(obj)}")
    else:
        print("\nNo objects detected in hybrid mode.")
    
    # Example: Process a video with both VLM-only and hybrid detection
    video_path = "output/dashcam_videos/Parking compliance Vantrue dashcam/20250602_065600_00002_T_A.MP4"
    
    # Check if video file exists, otherwise use sample image for demonstration
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Using sample image for video processing demonstration...")
        video_path = image_path  # Use the same sample image
    
    # Reuse existing pipeline and modify settings for different detection modes
    # Save original settings
    original_output_dir = pipeline.output_dir
    original_enable_traditional = pipeline.enable_traditional_detectors
    original_traditional_detectors = pipeline.traditional_detectors
    
    print("\n=== Video Processing with VLM-only Detection ===")
    # Configure pipeline for VLM-only detection
    pipeline.output_dir = "./qwen_video_results_vlm_only"
    pipeline.enable_traditional_detectors = False
    os.makedirs(pipeline.output_dir, exist_ok=True)
    
    # if video_path.endswith(('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV')):
    #     video_results_vlm = pipeline.process_video_frames(
    #         video_path, 
    #         extraction_method="scene_change", 
    #         use_sam_segmentation=True,
    #         save_results=True
    #     )
    #     print(f"VLM-only video processing: {video_results_vlm['summary']['frames_extracted']} frames extracted")
    #     print(f"Total objects detected: {video_results_vlm['summary']['total_objects_detected']}")
    # else:
    #     # Process as single image for demonstration
    #     video_results_vlm = pipeline.detect_objects(video_path, use_sam_segmentation=True, save_results=True)
    #     print(f"VLM-only detection (single image): {len(video_results_vlm['objects'])} objects")
    
    print("\n=== Video Processing with Hybrid Detection ===")
    # Configure pipeline for hybrid detection
    pipeline.output_dir = "./qwen_video_results_hybrid"
    pipeline.enable_traditional_detectors = True
    pipeline.traditional_detectors = ['yolo', 'detr']
    os.makedirs(pipeline.output_dir, exist_ok=True)
    
    if video_path.endswith(('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV')):
        video_results_hybrid = pipeline.process_video_frames(
            video_path, 
            extraction_method="scene_change", 
            use_sam_segmentation=True,
            save_results=True
        )
        print(f"Hybrid video processing: {video_results_hybrid['summary']['frames_extracted']} frames extracted")
        print(f"Total objects detected: {video_results_hybrid['summary']['total_objects_detected']}")
        print(f"Traditional detectors used: {video_results_hybrid['summary'].get('traditional_detectors_used', 0)}")
    else:
        # Process as single image for demonstration
        video_results_hybrid = pipeline.detect_objects_hybrid(video_path, use_sam_segmentation=True, save_results=True)
        print(f"Hybrid detection (single image): {len(video_results_hybrid['objects'])} objects")
        print(f"Traditional detectors used: {video_results_hybrid['traditional_detectors_used']}")
    
    print("\n=== Video Processing Results Comparison ===")
    print(f"VLM-only results saved to: ./qwen_video_results_vlm_only")
    print(f"Hybrid results saved to: ./qwen_video_results_hybrid")
    print("Results are saved in separate directories for easy comparison.")
    
    # Restore original pipeline settings
    pipeline.output_dir = original_output_dir
    pipeline.enable_traditional_detectors = original_enable_traditional
    pipeline.traditional_detectors = original_traditional_detectors
    
    # Example: Process multiple images
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # batch_results = pipeline.detect_objects_batch(image_paths)
    
    print("QwenObjectDetectionPipeline initialized successfully!")
    print(f"Output directory: {pipeline.output_dir}")
    print(f"SAM segmentation enabled: {pipeline.enable_sam}")
    print(f"Traditional detectors enabled: {pipeline.enable_traditional_detectors}")
    print(f"Available traditional detectors: {len(pipeline.traditional_detectors)}")
    print("\nTo use the pipeline:")
    print("1. VLM-only detection: pipeline.detect_objects('path/to/image.jpg')")
    print("2. Hybrid detection: pipeline.detect_objects_hybrid('path/to/image.jpg')")
    print("3. With SAM segmentation: pipeline.detect_objects_hybrid('path/to/image.jpg', use_sam_segmentation=True)")
    print("4. Video processing: pipeline.process_video_frames('path/to/video.mp4')")
    print("5. Multiple images: pipeline.detect_objects_batch(['img1.jpg', 'img2.jpg'])")


if __name__ == "__main__":
    main()
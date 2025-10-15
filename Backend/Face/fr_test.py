"""
Standalone Face Recognition System (No Django/React)

This script runs in two modes:
1) Register faces from an images directory and save embeddings to disk
2) Run real-time recognition from webcam (or a video file), recognizing multiple faces

Core design goals:
- Minimal mandatory dependencies (OpenCV + NumPy)
- Optional, more accurate pipeline if `face_recognition` (dlib) is installed
- Graceful fallback to an ORB-based embedding + matching when dlib is not available
- No network required at runtime

Embeddings format on disk (embeddings.pkl):
{
    'version': 2,
    'method': 'fr' | 'orb',
    'data': { name: [embedding, ...] }
}
Where:
- method 'fr': each embedding is a numpy float64 (128,) vector from face_recognition
- method 'orb': each embedding is a numpy uint8 (N, 32) matrix of ORB descriptors for a single reference face crop

Notes:
- Haar cascade is used for face detection as a default (ships with OpenCV). If you have a YOLO face model available, you can adapt the code to load it.
- DeepSORT is optional; a simple centroid tracker is provided.
"""

import cv2
import numpy as np
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import traceback

# Core libraries (required)
try:
    import face_recognition
    HAVE_FACE_RECOGNITION = True
except ImportError:
    print("WARNING: face_recognition not installed. Install with: pip install face_recognition")
    HAVE_FACE_RECOGNITION = False

# Optional libraries for better performance
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    print("INFO: YOLOv8 not available. Using OpenCV face detection instead.")
    print("     For better performance: pip install ultralytics")
    HAVE_YOLO = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAVE_DEEPSORT = True
except ImportError:
    print("INFO: DeepSORT not available. Using simple tracking.")
    print("     For better tracking: pip install deep-sort-realtime")
    HAVE_DEEPSORT = False

# ============================================================================
# Utilities
# ============================================================================

def ensure_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def nms_boxes(detections: List[List[int]], iou_thresh: float = 0.45) -> List[List[int]]:
    """Non-Maximum Suppression for [x,y,w,h,conf] boxes.
    Returns filtered detections in the same format.
    """
    if not detections:
        return []
    boxes = np.array([[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in detections], dtype=float)
    scores = np.array([d[4] if len(d) > 4 else 0.9 for d in detections], dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]


class HaarFaceDetector:
    """Lightweight face detector using OpenCV Haar cascade."""
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame):
        gray = ensure_gray(frame)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        # Normalize return to list of [x, y, w, h]
        try:
            if isinstance(faces, np.ndarray):
                if faces.size == 0:
                    return []
                return faces.astype(int).tolist()
            # Some OpenCV builds may return tuples/lists
            if hasattr(faces, '__len__'):
                if len(faces) == 0:
                    return []
                return [list(map(int, r)) for r in faces]
        except TypeError:
            # Guard against unexpected scalar returns
            return []
        except Exception:
            return []
        return []


class ORBEncoder:
    """ORB-based embedding encoder usable without dlib.

    We store ORB descriptors of a face crop as the embedding. Matching is done
    using BFMatcher with Hamming distance and a ratio test.
    """
    def __init__(self, n_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def encode(self, face_bgr) -> Optional[np.ndarray]:
        gray = ensure_gray(face_bgr)
        # Normalize contrast to improve keypoint detection
        try:
            gray = cv2.equalizeHist(gray)
        except Exception:
            pass
        # Resize to a consistent size to stabilize descriptor count
        try:
            h, w = gray.shape[:2]
            if min(h, w) > 0:
                scale = 160 / max(h, w)
                if scale != 1.0:
                    gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
        except Exception:
            pass
        kps, des = self.orb.detectAndCompute(gray, None)
        return des  # shape (N, 32) uint8 or None

    def compare(self, des_query: Optional[np.ndarray], des_ref_list: List[np.ndarray]) -> float:
        """Return a similarity score in [0,1] where higher is better.
        We compute the best score among reference descriptor sets.
        The score is the ratio of good matches (Lowe's ratio test) clipped to [0,1].
        """
        if des_query is None or len(des_query) == 0:
            return 0.0
        best = 0.0
        for des_ref in des_ref_list:
            if des_ref is None or len(des_ref) == 0:
                continue
            # k-NN matches
            matches = self.matcher.knnMatch(des_query, des_ref, k=2)
            good = 0
            try:
                for m_n in matches:
                    if not hasattr(m_n, '__len__') or len(m_n) != 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good += 1
            except TypeError:
                # In case matches is not iterable as expected
                good = 0
            # Normalize by number of keypoints to get a score [0,1]
            denom = max(len(des_query), 1)
            score = min(good / denom, 1.0)
            if score > best:
                best = score
        return float(best)


# ============================================================================
# PROGRAM 1: EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """Generate and store face embeddings from enrollment images"""
    
    def __init__(self, storage_path: str = "face_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.metadata_file = self.storage_path / "metadata.json"
        
    def register_faces(self, images_dir: str, model: str = "hog", method: str = "auto"):
        """
        Scan directory and generate embeddings for all faces found.
        
        Directory structure:
        images_dir/
            person1/
                image1.jpg
                image2.jpg
            person2/
                image1.jpg
        
        Or flat structure with filenames as person names:
        images_dir/
            person1.jpg
            person2.jpg
        """
        # Decide embedding method
        chosen_method = method
        if method == "auto":
            chosen_method = "fr" if HAVE_FACE_RECOGNITION else "orb"
        elif method not in ("fr", "orb"):
            print(f"ERROR: Unknown method '{method}'. Use 'auto' | 'fr' | 'orb'.")
            return False

        if chosen_method == "fr" and not HAVE_FACE_RECOGNITION:
            print("ERROR: face_recognition not installed. Install it or use method='orb'.")
            return False
            
        detector = HaarFaceDetector()
        orb = ORBEncoder() if chosen_method == "orb" else None

        embeddings_data = {}
        metadata = {}
        
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"ERROR: Directory {images_dir} not found")
            return False
        
        # Collect all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_path.rglob(ext))
        
        if not image_files:
            print(f"ERROR: No images found in {images_dir}")
            return False
        
        print(f"Found {len(image_files)} image(s) to process...")
        
        for img_path in image_files:
            # Determine person name from folder or filename
            if img_path.parent != images_path:
                person_name = img_path.parent.name
            else:
                person_name = img_path.stem
            
            print(f"Processing: {person_name} - {img_path.name}")
            
            # Load image (BGR for OpenCV ops)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  ⚠ Could not read image {img_path.name}")
                continue

            # Detect faces (use face_recognition if available and method='fr')
            face_bbox = None
            if HAVE_FACE_RECOGNITION and chosen_method == "fr":
                image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(image_rgb, model=model)
                if face_locations:
                    # (top, right, bottom, left) -> (x,y,w,h)
                    top, right, bottom, left = face_locations[0]
                    face_bbox = (left, top, right-left, bottom-top)
            else:
                dets = detector.detect(img_bgr)
                if dets:
                    # Use first detection (Haar has no confidence)
                    face_bbox = dets[0]

            if face_bbox is None:
                print(f"  ⚠ No face found in {img_path.name}")
                continue

            x, y, w, h = face_bbox
            # Expand crop slightly to include full face context
            mx = int(0.15 * w)
            my = int(0.20 * h)
            x0 = max(0, x - mx)
            y0 = max(0, y - my)
            x1 = min(img_bgr.shape[1], x + w + mx)
            y1 = min(img_bgr.shape[0], y + h + my)
            face_crop = img_bgr[y0:y1, x0:x1]
            if face_crop.size == 0:
                print(f"  ⚠ Invalid face crop in {img_path.name}")
                continue

            # Generate embedding
            if chosen_method == "fr":
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                locs = [(0, face_rgb.shape[1], face_rgb.shape[0], 0)]  # full crop
                encs = face_recognition.face_encodings(face_rgb, known_face_locations=locs)
                if not encs:
                    print(f"  ⚠ Could not generate embedding for {img_path.name}")
                    continue
                embedding = encs[0]
            else:  # ORB
                embedding = orb.encode(face_crop)
                if embedding is None or len(embedding) == 0:
                    print(f"  ⚠ Could not compute ORB descriptors for {img_path.name}")
                    continue

            # Store
            embeddings_data.setdefault(person_name, []).append(embedding)
            metadata.setdefault(person_name, []).append({
                'source_file': str(img_path),
                'bbox': [int(x), int(y), int(w), int(h)]
            })
            
            print(f"  ✓ Registered face for {person_name}")
        
        # Save to disk
        if embeddings_data:
            payload = {
                'version': 2,
                'method': chosen_method,
                'data': embeddings_data,
            }
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(payload, f)

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            total_people = len(embeddings_data)
            total_embeddings = sum(len(embs) for embs in embeddings_data.values())

            print(f"\n✅ Successfully registered {total_people} people with {total_embeddings} total embeddings")
            print(f"Method: {chosen_method}")
            print(f"Data saved to: {self.storage_path}")
            return True
        else:
            print("\n❌ No faces were successfully registered")
            return False
    
    def load_embeddings(self) -> Tuple[Dict, Dict]:
        """Load saved embeddings and metadata"""
        if not self.embeddings_file.exists():
            return {}, {}

        with open(self.embeddings_file, 'rb') as f:
            payload = pickle.load(f)

        # Backward compatibility: older files stored a plain dict of name->list[vec]
        if isinstance(payload, dict) and 'data' not in payload:
            embeddings = {'version': 1, 'method': 'fr' if HAVE_FACE_RECOGNITION else 'orb', 'data': payload}
        else:
            embeddings = payload

        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

        return embeddings, metadata

# ============================================================================
# PROGRAM 2: REAL-TIME RECOGNITION SYSTEM
# ============================================================================

class SimpleTracker:
    """Simple centroid-based tracker for when DeepSORT is not available"""
    
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark existing objects as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects
        
        input_centroids = []
        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for i, centroid in enumerate(input_centroids):
                self.objects[self.next_id] = detections[i]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = []
            
            for obj_id in object_ids:
                x, y, w, h = self.objects[obj_id]
                cx = x + w // 2
                cy = y + h // 2
                object_centroids.append((cx, cy))
            
            # Simple nearest neighbor matching
            used_detections = set()
            for i, obj_id in enumerate(object_ids):
                if len(used_detections) == len(input_centroids):
                    break
                    
                ox, oy = object_centroids[i]
                min_dist = float('inf')
                min_j = -1
                
                for j, (cx, cy) in enumerate(input_centroids):
                    if j in used_detections:
                        continue
                    dist = np.sqrt((ox - cx)**2 + (oy - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j
                
                if min_j != -1 and min_dist < 100:  # Distance threshold
                    self.objects[obj_id] = detections[min_j]
                    self.disappeared[obj_id] = 0
                    used_detections.add(min_j)
                else:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
            
            # Register new detections
            for j, centroid in enumerate(input_centroids):
                if j not in used_detections:
                    self.objects[self.next_id] = detections[j]
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
        
        return self.objects

class FaceRecognitionSystem:
    """Real-time face recognition with multiple detection methods"""
    
    def __init__(self, storage_path: str = "face_data"):
        self.storage_path = Path(storage_path)

        # Load known faces
        self.known_embeddings = {}
        self.known_metadata = {}
        self.method = 'fr'  # default, will be replaced by loaded file
        self.load_known_faces()
        
        # Initialize detectors based on availability
        self.yolo = None
        self.yolo_is_face = False
        if HAVE_YOLO:
            try:
                # Prefer a face-specific model if present
                script_dir = Path(__file__).resolve().parent
                candidates = [
                    script_dir / 'yolov8n-face.pt',
                    script_dir / 'models' / 'yolov8n-face.pt',
                    self.storage_path / 'models' / 'yolov8n-face.pt',
                ]
                face_weight = next((str(p) for p in candidates if p.exists()), None)
                if face_weight:
                    self.yolo = YOLO(face_weight)
                    self.yolo_is_face = True
                    print(f"✓ YOLOv8 face model loaded: {face_weight}")
                else:
                    # Fall back to generic COCO model, but we will not use it for face detection
                    self.yolo = YOLO('yolov8n.pt')
                    self.yolo_is_face = False
                    print("ℹ YOLOv8 COCO model loaded (person). For better face detection, place 'yolov8n-face.pt' in Face/models/")
            except Exception as e:
                print(f"Failed to initialize YOLO: {e}")
                self.yolo = None
                self.yolo_is_face = False

        # Try to initialize OpenCV DNN face detector (ResNet SSD)
        self.dnn = None
        try:
            script_dir = Path(__file__).resolve().parent
            proto_candidates = [
                script_dir / 'models' / 'deploy.prototxt',
                self.storage_path / 'models' / 'deploy.prototxt',
                script_dir / 'deploy.prototxt',
            ]
            model_candidates = [
                script_dir / 'models' / 'res10_300x300_ssd_iter_140000.caffemodel',
                self.storage_path / 'models' / 'res10_300x300_ssd_iter_140000.caffemodel',
                script_dir / 'res10_300x300_ssd_iter_140000.caffemodel',
            ]
            proto = next((str(p) for p in proto_candidates if p.exists()), None)
            model = next((str(p) for p in model_candidates if p.exists()), None)
            if proto and model:
                self.dnn = cv2.dnn.readNetFromCaffe(proto, model)
                # Try to use OpenCL if available for speed
                try:
                    self.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                except Exception:
                    try:
                        self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    except Exception:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              =
                        pass
                print("✓ OpenCV DNN face detector initialized")
        except Exception as e:
            print(f"INFO: DNN face detector not initialized: {e}")
        
        # Initialize tracker
        if HAVE_DEEPSORT:
            # Slightly stricter IOU to reduce duplicate tracks
            self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.5)
            print("✓ DeepSORT tracker initialized")
        else:
            self.tracker = SimpleTracker(max_disappeared=30)
            print("✓ Simple tracker initialized")

        # Face detector (Prefer DNN, then YOLO face, fallback to Haar)
        self.detector = HaarFaceDetector()

        # ORB encoder for fallback recognition
        self.orb_encoder = ORBEncoder()
        
        # Recognition settings
        self.recognition_threshold = 0.6  # Adjust for strictness
        self.fps_counter = {'time': time.time(), 'frames': 0, 'fps': 0}
    
    def load_known_faces(self):
        """Load pre-computed embeddings"""
        embeddings_file = self.storage_path / "embeddings.pkl"
        metadata_file = self.storage_path / "metadata.json"

        if not embeddings_file.exists():
            print("⚠ No embeddings found. Please run registration first.")
            return

        with open(embeddings_file, 'rb') as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and 'data' in payload:
            self.method = payload.get('method', 'fr')
            self.known_embeddings = payload['data']
        else:
            # Backward compatibility (assume fr)
            self.method = 'fr'
            self.known_embeddings = payload

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.known_metadata = json.load(f)

        total_people = len(self.known_embeddings)
        total_embeddings = sum(len(embs) for embs in self.known_embeddings.values())
        print(f"✓ Loaded {total_people} people with {total_embeddings} embeddings (method: {self.method})")
    
    def detect_faces_yolo(self, frame):
        """Detect faces using YOLO (if available)"""
        # NOTE: Default YOLOv8 COCO model detects 'person', not 'face'.
        # If you have a face model (e.g., yolov8n-face.pt), place it and change the initializer above.
        detections = []
        if self.yolo is not None and self.yolo_is_face:
            results = self.yolo(frame, verbose=False)[0]
            for box in results.boxes:
                # Safely extract scalar confidence to avoid NumPy deprecation warnings
                try:
                    conf_tensor = getattr(box, 'conf', None)
                    conf = float(conf_tensor.item()) if conf_tensor is not None else 0.0
                except Exception:
                    conf = 0.0
                if conf < 0.4:
                    continue
                # Safely extract coordinates regardless of tensor/ndarray shape
                try:
                    xyxy = box.xyxy
                    if hasattr(xyxy, 'cpu'):
                        xyxy = xyxy.cpu().numpy()
                    xyxy = np.array(xyxy).reshape(-1, 4)[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                except Exception:
                    continue
                detections.append([x1, y1, int(x2 - x1), int(y2 - y1), conf])
        return detections

    def detect_faces_dnn(self, frame, conf_thresh: float = 0.6):
        """Detect faces using OpenCV DNN ResNet SSD."""
        detections = []
        if self.dnn is None:
            return detections
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.dnn.setInput(blob)
        out = self.dnn.forward()
        # out shape: [1, 1, N, 7] -> [batch, class, detections, [id, classId, conf, x1, y1, x2, y2]]
        try:
            for i in range(out.shape[2]):
                confidence = float(out[0, 0, i, 2])
                if confidence < conf_thresh:
                    continue
                x1 = int(out[0, 0, i, 3] * w)
                y1 = int(out[0, 0, i, 4] * h)
                x2 = int(out[0, 0, i, 5] * w)
                y2 = int(out[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                detections.append([x1, y1, max(0, x2 - x1), max(0, y2 - y1), confidence])
        except Exception:
            pass
        return detections
    
    def detect_faces_opencv(self, frame):
        """Detect faces using Haar cascade."""
        try:
            return self.detector.detect(frame)
        except TypeError:
            # e.g., if underlying cascade returns an unexpected scalar
            return []
        except Exception as e:
            print(f"Detect error (OpenCV Haar): {e}")
            return []
    
    def recognize_face(self, face_image_bgr):
        """Recognize a single face crop by comparing with known embeddings.

        Returns: (name, confidence)
        - For 'fr' method, confidence = 1 - normalized_distance (approx)
        - For 'orb' method, confidence = ORB match score in [0,1]
        """
        if not self.known_embeddings:
            return "Unknown", 0.0

        if self.method == 'fr':
            if not HAVE_FACE_RECOGNITION:
                return "Unknown", 0.0
            face_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
            # full crop as the face region
            locs = [(0, face_rgb.shape[1], face_rgb.shape[0], 0)]
            encs = face_recognition.face_encodings(face_rgb, known_face_locations=locs)
            if not encs:
                return "Unknown", 0.0
            face_encoding = encs[0]

            best_name = "Unknown"
            best_dist = 1.0
            for name, known_list in self.known_embeddings.items():
                try:
                    known_arr = np.asarray(known_list)
                    if known_arr.ndim == 1:
                        known_arr = known_arr.reshape(1, -1)
                    distances = face_recognition.face_distance(known_arr, face_encoding)
                    if len(distances) == 0:
                        continue
                    d = float(np.min(distances))
                except Exception:
                    # In case stored vectors are malformed
                    continue
                if d < best_dist:
                    best_dist = d
                    best_name = name
            # Convert distance to pseudo-confidence
            conf = max(0.0, 1.0 - (best_dist / max(self.recognition_threshold, 1e-6)))
            if best_dist > self.recognition_threshold:
                best_name = "Unknown"
                conf = 0.0
            return best_name, float(conf)

        else:  # ORB
            des_q = self.orb_encoder.encode(face_image_bgr)
            best_name = "Unknown"
            best_score = 0.0
            for name, ref_list in self.known_embeddings.items():
                score = self.orb_encoder.compare(des_q, ref_list)
                if score > best_score:
                    best_score = score
                    best_name = name
            # Threshold for ORB score; tuneable. 0.28 works reasonably for many setups.
            if best_score < 0.28:
                return "Unknown", float(best_score)
            return best_name, float(best_score)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter['frames'] += 1
        current_time = time.time()
        time_diff = current_time - self.fps_counter['time']
        
        if time_diff > 1.0:
            self.fps_counter['fps'] = self.fps_counter['frames'] / time_diff
            self.fps_counter['frames'] = 0
            self.fps_counter['time'] = current_time
        
        return self.fps_counter['fps']
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        results = []
        height, width = frame.shape[:2]
        
        # Detect faces: prefer DNN > YOLO(face) > Haar
        if self.dnn is not None:
            detections = self.detect_faces_dnn(frame)
        elif HAVE_YOLO and self.yolo is not None and self.yolo_is_face:
            detections = self.detect_faces_yolo(frame)
        else:
            # Haar: no confidence, assign a modest one
            raw = self.detect_faces_opencv(frame)
            detections = [[x, y, w, h, 0.6] for (x, y, w, h) in raw]
        # Normalize detections to list of [x,y,w,h,conf]
        norm = []
        try:
            if isinstance(detections, (list, tuple, np.ndarray)):
                for d in detections:
                    try:
                        if len(d) == 5:
                            x, y, w, h, conf = d
                        else:
                            x, y, w, h = d
                            conf = 0.9
                        norm.append([int(x), int(y), int(w), int(h), float(conf)])
                    except Exception:
                        continue
            else:
                norm = []
        except Exception:
            norm = []
        # Apply NMS to reduce duplicate boxes
        detections = nms_boxes(norm, iou_thresh=0.45)
        
        # Update tracker
        if HAVE_DEEPSORT:
            # DeepSORT expects different format
            ds_detections = []
            for det in detections:
                x, y, w, h, conf = det if len(det) == 5 else (*det, 0.9)
                # Format: ([x, y, w, h], confidence, class)
                ds_detections.append(([int(x), int(y), int(w), int(h)], float(conf), 0))
            
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x, y = int(ltrb[0]), int(ltrb[1])
                w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
                
                # Extract face region
                # Expand ROI a bit for better recognition stability
                mx = int(0.12 * w)
                my = int(0.18 * h)
                x0 = max(0, x - mx)
                y0 = max(0, y - my)
                x1 = min(frame.shape[1], x + w + mx)
                y1 = min(frame.shape[0], y + h + my)
                face_roi = frame[y0:y1, x0:x1]
                if face_roi.size > 0:
                    name, conf = self.recognize_face(face_roi)
                    results.append({
                        'id': track_id,
                        'bbox': (x, y, w, h),
                        'name': name,
                        'confidence': conf
                    })
        else:
            # Simple tracker
            try:
                tracked_objects = self.tracker.update(detections)
            except TypeError:
                # Guard against wrong detection type
                tracked_objects = self.tracker.update([])
            
            for obj_id, (x, y, w, h) in tracked_objects.items():
                # Extract face region
                face_roi = frame[max(0,y):y+h, max(0,x):x+w]
                if face_roi.size > 0:
                    try:
                        name, conf = self.recognize_face(face_roi)
                    except Exception:
                        name, conf = "Unknown", 0.0
                    results.append({
                        'id': obj_id,
                        'bbox': (x, y, w, h),
                        'name': name,
                        'confidence': conf
                    })
        
        return results
    
    def draw_results(self, frame, results, fps):
        """Draw bounding boxes and labels on frame"""
        if not isinstance(results, (list, tuple)):
            results = []
        for result in results:
            if not isinstance(result, dict):
                continue
            x, y, w, h = result['bbox']
            name = result['name']
            track_id = result['id']
            conf = result['confidence']
            
            # Color based on recognition status
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label
            label = f"ID:{track_id} {name}"
            if name != "Unknown":
                label += f" ({conf:.2f})"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = y - 10 if y - 10 > 0 else y + h + 20
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5),
                         (x + label_size[0], label_y + 5), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw instructions
        info_text = "Press 'q' to quit | 's' to save frame | 'r' to reload embeddings"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(self, camera_index=0, video_path: Optional[str] = None):
        """Run real-time recognition on webcam"""
        # If a video file is provided, open it directly
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file. Check the path and try again.")
            backend_name = "FILE"
        else:
            # Probe multiple Windows backends for reliability, similar to fixcamera.py
            backend_candidates = [
                ("MSMF", getattr(cv2, 'CAP_MSMF', cv2.CAP_ANY)),
                ("DSHOW", getattr(cv2, 'CAP_DSHOW', cv2.CAP_ANY)),
                ("ANY", cv2.CAP_ANY),
                ("VFW", getattr(cv2, 'CAP_VFW', cv2.CAP_ANY)),
            ]

            cap = None
            backend_name = None
            for name, code in backend_candidates:
                try:
                    tmp = cv2.VideoCapture(camera_index, code)
                    if tmp.isOpened():
                        # Validate we can read at least one frame
                        ok, frm = tmp.read()
                        if ok and frm is not None:
                            cap = tmp
                            backend_name = name
                            break
                    tmp.release()
                except Exception:
                    continue

            if cap is None or not cap.isOpened():
                raise RuntimeError("Could not open camera. Try a different index (0/1/2) or close other apps using the camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("FACE RECOGNITION SYSTEM RUNNING")
        print("="*50)
        print("Controls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  r - Reload embeddings")
        print("  SPACE - Pause/Resume")
        print("="*50)
        if not video_path and backend_name:
            print(f"Using camera backend: {backend_name}")
        
        paused = False
        frame_count = 0
        display_frame = None
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: 
                    print("Failed to grab frame")
                    break
                
                # Process every 2nd frame for better performance
                if frame_count % 2 == 0:
                    try:
                        results = self.process_frame(frame)
                    except Exception as e:
                        print(f"Process frame error: {e}")
                        traceback.print_exc()
                        results = []
                    self.last_results = results
                else:
                    results = getattr(self, 'last_results', [])
                
                frame_count += 1
                
                # Update FPS
                fps = self.update_fps()
                
                # Draw results
                display_frame = self.draw_results(frame.copy(), results, fps)
            else:
                if display_frame is None:
                    # If paused before first frame processed, just show a black frame
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Show frame
            cv2.imshow('Face Recognition System', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                to_save = display_frame if display_frame is not None else frame
                cv2.imwrite(filename, to_save)
                print(f"✓ Frame saved as {filename}")
            elif key == ord('r'):
                self.load_known_faces()
                print("✓ Embeddings reloaded")
            elif key == 32:  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
        print(3)
        cap.release()
        cv2.destroyAllWindows()

# ============================================================================
# MAIN TEST FUNCTIONS
# ============================================================================

def test_system():
    """Test the complete face recognition system"""
    
    print("\n" + "="*60)
    print("FACE RECOGNITION SYSTEM - STANDALONE TEST")
    print("="*60)
    
    # Info about pipeline
    if not HAVE_FACE_RECOGNITION:
        print("\nINFO: face_recognition not installed. Falling back to ORB-based recognition.")
        print("For higher accuracy, install dlib-backed pipeline later:")
        print("  pip install face-recognition  (requires dlib)")
    
    while True:
        print("\nSelect an option:")
        print("1. Register faces from directory")
        print("2. Run real-time recognition")
        print("3. Create test images (for demo)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Register faces
            images_dir = input("Enter path to images directory (or press Enter for 'enrollment_images'): ").strip()
            if not images_dir:
                images_dir = "enrollment_images"
            
            generator = EmbeddingGenerator()
            success = generator.register_faces(images_dir)
            
            if not success:
                print("\nTip: Create a directory structure like:")
                print("  enrollment_images/")
                print("    Alice/")
                print("      alice1.jpg")
                print("    Bob/")
                print("      bob1.jpg")
        
        elif choice == '2':
            # Run recognition
            system = FaceRecognitionSystem()
            
            if not system.known_embeddings:
                print("\n⚠ No registered faces found!")
                print("Please register faces first (Option 1)")
                continue
            
            try:
                system.run_webcam()
            except Exception as e:
                print(f"\n❌ Error running webcam: {e}")
                import traceback as _tb
                _tb.print_exc()
                print("Make sure your camera is connected and not being used by another application")
        
        elif choice == '3':
            # Create test structure
            print("\nCreating test directory structure...")
            test_dir = Path("enrollment_images")
            test_dir.mkdir(exist_ok=True)
            
            # Create sample folders
            (test_dir / "Alice").mkdir(exist_ok=True)
            (test_dir / "Bob").mkdir(exist_ok=True)
            
            print(f"✓ Created test directory: {test_dir.absolute()}")
            print("\nNow add images to:")
            print(f"  {test_dir / 'Alice'} / alice1.jpg, alice2.jpg, etc.")
            print(f"  {test_dir / 'Bob'} / bob1.jpg, bob2.jpg, etc.")
            print("\nThen run Option 1 to register these faces")
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    # Print system info
    print("\nSystem Check:")
    print(f"  face_recognition: {'✓' if HAVE_FACE_RECOGNITION else '✗ (fallback to ORB)'}")
    print(f"  YOLOv8:          {'✓' if HAVE_YOLO else '✗ (optional)'}")
    print(f"  DeepSORT:        {'✓' if HAVE_DEEPSORT else '✗ (optional)'}")
    print(f"  OpenCV:          ✓")

    # Run test system
    test_system()
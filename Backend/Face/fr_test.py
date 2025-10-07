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


class HaarFaceDetector:
    """Lightweight face detector using OpenCV Haar cascade."""
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame):
        gray = ensure_gray(frame)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces.tolist() if len(faces) > 0 else []


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
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good += 1
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
                    face_bbox = dets[0]

            if face_bbox is None:
                print(f"  ⚠ No face found in {img_path.name}")
                continue

            x, y, w, h = face_bbox
            face_crop = img_bgr[max(0,y):y+h, max(0,x):x+w]
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
        if HAVE_YOLO:
            try:
                self.yolo = YOLO('yolov8n.pt')  # Using nano for faster inference
                print("✓ YOLOv8 initialized")
            except Exception as e:
                print(f"Failed to initialize YOLO: {e}")
                self.yolo = None
        
        # Initialize tracker
        if HAVE_DEEPSORT:
            self.tracker = DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)
            print("✓ DeepSORT tracker initialized")
        else:
            self.tracker = SimpleTracker(max_disappeared=30)
            print("✓ Simple tracker initialized")
        
        # Face detector (Haar by default)
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
        if self.yolo is not None:
            results = self.yolo(frame, verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf.cpu().numpy())
                if conf < 0.4:
                    continue
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                detections.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        return detections
    
    def detect_faces_opencv(self, frame):
        """Detect faces using Haar cascade."""
        return self.detector.detect(frame)
    
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
                    distances = face_recognition.face_distance(known_list, face_encoding)
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
        
        # Detect faces/persons
        if HAVE_YOLO and self.yolo is not None:
            detections = self.detect_faces_yolo(frame)
        else:
            detections = self.detect_faces_opencv(frame)
        
        # Update tracker
        if HAVE_DEEPSORT:
            # DeepSORT expects different format
            ds_detections = []
            for (x, y, w, h) in detections:
                ds_detections.append([x, y, w, h, 0.9])  # Add confidence
            
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x, y = int(ltrb[0]), int(ltrb[1])
                w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
                
                # Extract face region
                face_roi = frame[max(0,y):y+h, max(0,x):x+w]
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
            tracked_objects = self.tracker.update(detections)
            
            for obj_id, (x, y, w, h) in tracked_objects.items():
                # Extract face region
                face_roi = frame[max(0,y):y+h, max(0,x):x+w]
                if face_roi.size > 0:
                    name, conf = self.recognize_face(face_roi)
                    results.append({
                        'id': obj_id,
                        'bbox': (x, y, w, h),
                        'name': name,
                        'confidence': conf
                    })
        
        return results
    
    def draw_results(self, frame, results, fps):
        """Draw bounding boxes and labels on frame"""
        for result in results:
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
        cap = cv2.VideoCapture(video_path if video_path else camera_index)
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
                    results = self.process_frame(frame)
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
                cv2.imwrite(filename, frame)
                print(f"✓ Frame saved as {filename}")
            elif key == ord('r'):
                self.load_known_faces()
                print("✓ Embeddings reloaded")
            elif key == 32:  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
        
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
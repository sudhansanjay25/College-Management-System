"""
InsightFace-based Attendance System
High accuracy face recognition using ArcFace model
Designed to work with folder structure: enrollment_images/Person Name/1.jpg, 2.jpg, etc.
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
import json

class InsightFaceAttendanceSystem:
    def __init__(self, 
                 enrollment_folder='enrollment_images',
                 data_file='face_data/face_database.pkl',
                 threshold=0.5):
        """
        Initialize InsightFace attendance system
        
        Args:
            enrollment_folder: Path to folder containing person folders with images
            data_file: Path to save/load face embeddings
            threshold: Similarity threshold (0.3-0.6, lower = stricter)
        """
        print("🚀 Initializing InsightFace...")
        
        # Initialize InsightFace
        self.app = FaceAnalysis(
            name='buffalo_l',  # buffalo_l is the most accurate model
            providers=['CPUExecutionProvider']  # Use GPU if available: ['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.enrollment_folder = enrollment_folder
        self.data_file = data_file
        self.threshold = threshold
        
        # Database
        self.face_database = {
            'embeddings': [],
            'names': [],
            'person_ids': []
        }
        
        print("✅ InsightFace initialized successfully!")
        print(f"   Model: buffalo_l (ArcFace)")
        print(f"   Threshold: {threshold}")
    
    def get_face_embedding(self, image):
        """
        Get face embedding from image
        Returns: (embedding, bbox) or (None, None) if no face found
        """
        # Detect faces
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face (assuming it's the main subject)
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        return face.embedding, face.bbox
    
    def load_enrollment_images(self):
        """
        Load all images from enrollment_images folder structure
        Expected structure:
            enrollment_images/
                Person Name 1/
                    1.jpg
                    2.jpg
                    ...
                Person Name 2/
                    1.jpg
                    2.jpg
                    ...
        """
        print(f"\n📂 Loading enrollment images from '{self.enrollment_folder}'...")
        
        if not os.path.exists(self.enrollment_folder):
            print(f"❌ Folder '{self.enrollment_folder}' not found!")
            return False
        
        person_folders = [f for f in Path(self.enrollment_folder).iterdir() if f.is_dir()]
        
        if not person_folders:
            print(f"❌ No person folders found in '{self.enrollment_folder}'!")
            return False
        
        total_persons = 0
        total_images = 0
        
        for person_folder in person_folders:
            person_name = person_folder.name
            person_id = person_name.lower().replace(' ', '_')
            
            print(f"\n👤 Processing: {person_name}")
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(person_folder.glob(ext))
            
            if not image_files:
                print(f"   ⚠️ No images found for {person_name}")
                continue
            
            embeddings_for_person = []
            
            for img_path in image_files:
                # Read image
                image = cv2.imread(str(img_path))
                
                if image is None:
                    print(f"   ⚠️ Could not read {img_path.name}")
                    continue
                
                # Get embedding
                embedding, bbox = self.get_face_embedding(image)
                
                if embedding is not None:
                    embeddings_for_person.append(embedding)
                    print(f"   ✅ Processed {img_path.name}")
                    total_images += 1
                else:
                    print(f"   ⚠️ No face detected in {img_path.name}")
            
            # Average all embeddings for this person
            if embeddings_for_person:
                avg_embedding = np.mean(embeddings_for_person, axis=0)
                
                self.face_database['embeddings'].append(avg_embedding)
                self.face_database['names'].append(person_name)
                self.face_database['person_ids'].append(person_id)
                
                total_persons += 1
                print(f"   ✅ Added {person_name} with {len(embeddings_for_person)} samples")
            else:
                print(f"   ❌ No valid faces found for {person_name}")
        
        print(f"\n✅ Enrollment complete!")
        print(f"   Total persons: {total_persons}")
        print(f"   Total images processed: {total_images}")
        
        return total_persons > 0
    
    def save_database(self):
        """Save face database to file"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.face_database, f)
        
        print(f"\n💾 Database saved to {self.data_file}")
        print(f"   Persons in database: {len(self.face_database['names'])}")
    
    def load_database(self):
        """Load face database from file"""
        if not os.path.exists(self.data_file):
            print(f"⚠️ Database file not found: {self.data_file}")
            return False
        
        with open(self.data_file, 'rb') as f:
            self.face_database = pickle.load(f)
        
        print(f"✅ Database loaded from {self.data_file}")
        print(f"   Persons in database: {len(self.face_database['names'])}")
        return True
    
    def recognize_face(self, image):
        """
        Recognize face in image
        Returns: (name, person_id, confidence, bbox) or (None, None, 0, None)
        """
        # Get embedding for the face
        embedding, bbox = self.get_face_embedding(image)
        
        if embedding is None:
            return None, None, 0.0, None
        
        if not self.face_database['embeddings']:
            return "Unknown", None, 0.0, bbox
        
        # Calculate similarity with all known faces
        embeddings_array = np.array(self.face_database['embeddings'])
        
        # Cosine similarity
        similarities = np.dot(embeddings_array, embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(embedding)
        )
        
        # Get best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Convert similarity to confidence (0-100%)
        confidence = best_similarity * 100
        
        # Check if match is good enough
        if best_similarity >= self.threshold:
            name = self.face_database['names'][best_match_idx]
            person_id = self.face_database['person_ids'][best_match_idx]
        else:
            name = "Unknown"
            person_id = None
        
        return name, person_id, confidence, bbox
    
    def run_webcam_recognition(self):
        """
        Run real-time face recognition from webcam
        """
        if not self.face_database['embeddings']:
            print("❌ No faces in database! Load or enroll faces first.")
            return
        
        print("\n🎥 Starting webcam recognition...")
        print("Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'a' to mark attendance")
        print("   - Press 't' to adjust threshold")
        
        # Open webcam
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Attendance log
        attendance_log = {}
        
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame for performance
        last_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Process face recognition
                if frame_count % process_every_n_frames == 0:
                    # Get all faces in frame
                    faces = self.app.get(frame)
                    last_results = []
                    
                    for face in faces:
                        embedding = face.embedding
                        bbox = face.bbox.astype(int)
                        
                        # Calculate similarity with database
                        if self.face_database['embeddings']:
                            embeddings_array = np.array(self.face_database['embeddings'])
                            similarities = np.dot(embeddings_array, embedding) / (
                                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(embedding)
                            )
                            
                            best_match_idx = np.argmax(similarities)
                            best_similarity = similarities[best_match_idx]
                            confidence = best_similarity * 100
                            
                            if best_similarity >= self.threshold:
                                name = self.face_database['names'][best_match_idx]
                                person_id = self.face_database['person_ids'][best_match_idx]
                            else:
                                name = "Unknown"
                                person_id = None
                            
                            last_results.append((name, person_id, confidence, bbox))
                
                # Draw results
                for name, person_id, confidence, bbox in last_results:
                    x1, y1, x2, y2 = bbox
                    
                    # Color based on recognition
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green
                        label = f"{name} ({confidence:.1f}%)"
                    else:
                        color = (0, 0, 255)  # Red
                        label = "Unknown"
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    
                    # Draw text
                    cv2.putText(frame, label, (x1 + 6, y2 - 6),
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Display info
                info_text = f"Threshold: {self.threshold:.2f} | Known: {len(self.face_database['names'])} | Detected: {len(last_results)}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('InsightFace Attendance System', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'screenshot_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('a'):
                    # Mark attendance for all recognized faces
                    for name, person_id, confidence, bbox in last_results:
                        if person_id:
                            if person_id not in attendance_log:
                                attendance_log[person_id] = {
                                    'name': name,
                                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'confidence': confidence
                                }
                                print(f"✅ Attendance marked: {name} ({confidence:.1f}%)")
                            else:
                                print(f"⚠️ {name} already marked present")
                elif key == ord('t'):
                    try:
                        new_threshold = float(input("\nEnter new threshold (0.3-0.6): "))
                        self.threshold = max(0.2, min(0.8, new_threshold))
                        print(f"✅ Threshold set to {self.threshold}")
                    except:
                        print("❌ Invalid input")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save attendance log
            if attendance_log:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f'attendance_log_{timestamp}.json'
                with open(log_file, 'w') as f:
                    json.dump(attendance_log, f, indent=4)
                print(f"\n📋 Attendance log saved: {log_file}")
                print(f"   Total present: {len(attendance_log)}")


def main():
    """Main application"""
    print("=" * 70)
    print("InsightFace Attendance System")
    print("High Accuracy Face Recognition")
    print("=" * 70)
    
    # Initialize system
    system = InsightFaceAttendanceSystem(
        enrollment_folder='enrollment_images',
        data_file='face_data/face_database.pkl',
        threshold=0.45  # Adjust based on your needs (0.4-0.5 recommended)
    )
    
    while True:
        print("\n" + "=" * 70)
        print("Options:")
        print("1. Load enrollment images from folder")
        print("2. Save database")
        print("3. Load existing database")
        print("4. Start webcam recognition")
        print("5. Test with single image")
        print("6. View database info")
        print("7. Adjust threshold")
        print("8. Exit")
        print("=" * 70)
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            if system.load_enrollment_images():
                print("\n✅ Enrollment successful!")
                save = input("Save database now? (y/n): ").lower()
                if save == 'y':
                    system.save_database()
            else:
                print("\n❌ Enrollment failed!")
        
        elif choice == "2":
            if system.face_database['embeddings']:
                system.save_database()
            else:
                print("❌ No data to save! Load enrollment images first.")
        
        elif choice == "3":
            system.load_database()
        
        elif choice == "4":
            system.run_webcam_recognition()
        
        elif choice == "5":
            image_path = input("Enter image path: ")
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                name, person_id, confidence, bbox = system.recognize_face(image)
                
                if name:
                    print(f"\n✅ Recognized: {name}")
                    print(f"   Person ID: {person_id}")
                    print(f"   Confidence: {confidence:.2f}%")
                else:
                    print("\n❌ No face detected")
            else:
                print("❌ Image not found")
        
        elif choice == "6":
            print(f"\n📊 Database Info:")
            print(f"   Total persons: {len(system.face_database['names'])}")
            print(f"   Threshold: {system.threshold}")
            if system.face_database['names']:
                print(f"\n   Enrolled persons:")
                for i, name in enumerate(system.face_database['names'], 1):
                    print(f"      {i}. {name}")
        
        elif choice == "7":
            try:
                new_threshold = float(input("Enter new threshold (0.3-0.6 recommended): "))
                system.threshold = max(0.2, min(0.8, new_threshold))
                print(f"✅ Threshold set to {system.threshold}")
            except:
                print("❌ Invalid input")
        
        elif choice == "8":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
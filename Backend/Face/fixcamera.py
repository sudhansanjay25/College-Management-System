"""
Camera Access Fix - Try Multiple Backends
This script tries different backends to find the one that works with your camera
"""

import cv2
import face_recognition
import numpy as np

def try_camera_backends(camera_index=0):
    """
    Try different camera backends to find which one works
    """
    # List of backends to try (in order of preference for Windows)
    backends = [
        ("AUTO", cv2.CAP_ANY),
        ("MSMF", cv2.CAP_MSMF),      # Microsoft Media Foundation (Windows default)
        ("DSHOW", cv2.CAP_DSHOW),     # DirectShow
        ("VFW", cv2.CAP_VFW),         # Video for Windows (legacy)
    ]
    
    print("🔍 Testing different camera backends...\n")
    working_backend = None
    
    for name, backend in backends:
        print(f"Testing {name} backend...")
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ {name} backend works! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    working_backend = (name, backend)
                    cap.release()
                    break
                else:
                    print(f"⚠️ {name} backend opened but can't read frames")
            else:
                print(f"❌ {name} backend failed to open camera")
            
            cap.release()
        except Exception as e:
            print(f"❌ {name} backend error: {e}")
    
    if working_backend:
        print(f"\n✅ Best backend: {working_backend[0]}")
        return working_backend
    else:
        print("\n❌ No working backend found!")
        return None

def run_face_recognition_fixed(camera_index=0, backend=None):
    """
    Fixed face recognition with proper backend handling
    """
    print(f"\n🚀 Starting face recognition...")
    
    # Determine backend
    if backend is None:
        result = try_camera_backends(camera_index)
        if result is None:
            print("❌ Cannot find working camera backend")
            return
        backend_name, backend_code = result
    else:
        backend_name, backend_code = backend
    
    print(f"\n📹 Using {backend_name} backend")
    
    # Open camera with determined backend
    video_capture = cv2.VideoCapture(camera_index, backend_code)
    
    if not video_capture.isOpened():
        print("❌ Failed to open camera")
        return
    
    # Set camera properties
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press SPACE to toggle face detection\n")
    
    face_detection_enabled = True
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame
    
    try:
        while True:
            ret, frame = video_capture.read()
            
            if not ret or frame is None:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Face detection
            if face_detection_enabled and (frame_count % process_every_n_frames == 0):
                try:
                    # Resize for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
                    
                    # Draw boxes around faces
                    for (top, right, bottom, left) in face_locations:
                        # Scale back up
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # Add label background
                        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(display_frame, "Person Detected", (left + 6, bottom - 6),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display count
                    if len(face_locations) > 0:
                        cv2.putText(display_frame, f"Faces: {len(face_locations)}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                except Exception as e:
                    cv2.putText(display_frame, f"Detection Error", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Status text
            status = "ON" if face_detection_enabled else "OFF"
            color = (0, 255, 0) if face_detection_enabled else (0, 0, 255)
            cv2.putText(display_frame, f"Detection: {status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display frame
            cv2.imshow('Face Recognition - Press Q to quit', display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                filename = f'capture_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
            elif key == ord(' '):
                face_detection_enabled = not face_detection_enabled
                status = "enabled" if face_detection_enabled else "disabled"
                print(f"Face detection {status}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def simple_camera_test(camera_index=0):
    """
    Simple camera test without face recognition
    """
    print("🎥 Simple camera test (no face detection)")
    
    # Try MSMF first (Windows default)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print("Trying AUTO backend...")
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    print("✅ Camera opened! Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Cannot read frame")
            break
        
        cv2.imshow('Camera Test - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("Camera & Face Recognition Fix Tool")
    print("=" * 60)
    
    camera_index = 0
    
    print("\nOptions:")
    print("1. Simple camera test (recommended first)")
    print("2. Face recognition with auto-backend detection")
    print("3. Try all backends")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        simple_camera_test(camera_index)
    elif choice == "2":
        run_face_recognition_fixed(camera_index)
    elif choice == "3":
        try_camera_backends(camera_index)
    else:
        print("Invalid choice. Running simple test...")
        simple_camera_test(camera_index)

if __name__ == "__main__":
    main()
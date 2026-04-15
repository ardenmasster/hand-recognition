import cv2
import mediapipe as mp
import pyautogui
import time
import sys

class TwoHandPowerPointController:
    """
    Controls PowerPoint presentations using two-hand gesture recognition.
    
    Gesture mappings:
    - Left Fist + Right Point → Previous Slide (←)
    - Right Fist + Left Point → Next Slide (→)
    - Both Pinched → Start Presentation (F5)
    - Both Open → End Presentation (ESC)
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # seconds
        
    def is_hand_open(self, hand_landmarks):
        """Check if hand is open (fingers extended)."""
        # Get finger tips and their respective joints
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get palm base for reference
        wrist = hand_landmarks.landmark[0]
        
        # Check if all fingers are above palm (extended)
        fingers_extended = (
            index_tip.y < wrist.y - 0.1 and
            middle_tip.y < wrist.y - 0.1 and
            ring_tip.y < wrist.y - 0.1 and
            pinky_tip.y < wrist.y - 0.1
        )
        
        return fingers_extended
    
    def is_hand_pinched(self, hand_landmarks):
        """Check if hand is in pinch gesture (thumb and index finger close)."""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate distance between thumb and index
        distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                   (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        
        # If distance is small, it's pinched
        return distance < 0.05
    
    def is_hand_fist(self, hand_landmarks):
        """Check if hand is in fist gesture."""
        # In fist, all fingertips should be below knuckles
        index_tip = hand_landmarks.landmark[8]
        index_knuckle = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_knuckle = hand_landmarks.landmark[9]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_knuckle = hand_landmarks.landmark[13]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_knuckle = hand_landmarks.landmark[17]
        
        # All fingertips below knuckles indicates closed fist
        fist = (
            index_tip.y > index_knuckle.y and
            middle_tip.y > middle_knuckle.y and
            ring_tip.y > ring_knuckle.y and
            pinky_tip.y > pinky_knuckle.y
        )
        
        return fist
    
    def is_pointing(self, hand_landmarks):
        """Check if hand is pointing (index finger extended, others closed)."""
        index_tip = hand_landmarks.landmark[8]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_mcp = hand_landmarks.landmark[9]
        
        wrist = hand_landmarks.landmark[0]
        
        # Index finger should extend upward and be the highest point
        index_extended = index_tip.y < index_mcp.y and index_tip.y < wrist.y - 0.1
        
        # Other fingers closed
        middle_closed = middle_tip.y > middle_mcp.y
        
        return index_extended and middle_closed
    
    def get_hand_side(self, hand_landmarks, handedness_label):
        """
        Determine if hand is left or right.
        MediaPipe gives handedness, but also check hand center position.
        """
        return handedness_label.lower()

    def _is_frame_black(self, frame, threshold=12.0):
        """Return True when frame intensity is too low to be a usable camera image."""
        if frame is None or frame.size == 0:
            return True
        return float(frame.mean()) < threshold

    def _open_camera(self, camera_indexes=(0, 1, 2), warmup_frames=20):
        """Try several camera indexes/backends and return the first feed that is not black."""
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Default"),
        ]

        for camera_index in camera_indexes:
            for backend, backend_name in backends:
                cap = cv2.VideoCapture(camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Request a common preview size to reduce backend-dependent defaults.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                has_non_black_frame = False
                for _ in range(warmup_frames):
                    ret, frame = cap.read()
                    if ret and not self._is_frame_black(frame):
                        has_non_black_frame = True
                        break

                if has_non_black_frame:
                    print(f"✅ Camera opened on index {camera_index} using {backend_name}")
                    return cap, camera_index, backend_name

                print(f"⚠️ Camera index {camera_index} on {backend_name} returned black frames")
                cap.release()

        return None, None, None
    
    def process_frame(self, frame):
        """
        Process a single frame and detect gestures.
        Returns True if a gesture action was taken.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        action_taken = False
        current_time = time.time()
        
        # Check cooldown to prevent rapid repeated commands
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # We have both hands
            hand1_landmarks = results.multi_hand_landmarks[0]
            hand2_landmarks = results.multi_hand_landmarks[1]
            hand1_label = results.multi_handedness[0].classification[0].label
            hand2_label = results.multi_handedness[1].classification[0].label
            
            # Ensure we know which hand is which
            if hand1_label.lower() == 'left':
                left_hand = hand1_landmarks
                right_hand = hand2_landmarks
            else:
                left_hand = hand2_landmarks
                right_hand = hand1_landmarks
            
            # Check gesture combinations
            left_fist = self.is_hand_fist(left_hand)
            left_open = self.is_hand_open(left_hand)
            left_pinched = self.is_hand_pinched(left_hand)
            left_pointing = self.is_pointing(left_hand)
            
            right_fist = self.is_hand_fist(right_hand)
            right_open = self.is_hand_open(right_hand)
            right_pinched = self.is_hand_pinched(right_hand)
            right_pointing = self.is_pointing(right_hand)
            
            # Gesture 1: Left Fist + Right Point → Previous Slide (←)
            if left_fist and right_pointing:
                pyautogui.press('left')
                print("🔴 Detected: Left Fist + Right Point → Previous Slide")
                sys.stdout.flush()
                action_taken = True
            
            # Gesture 2: Right Fist + Left Point → Next Slide (→)
            elif right_fist and left_pointing:
                pyautogui.press('right')
                print("🔴 Detected: Right Fist + Left Point → Next Slide")
                sys.stdout.flush()
                action_taken = True
            
            # Gesture 3: Both Pinched → Start Presentation (F5)
            elif left_pinched and right_pinched:
                pyautogui.press('f5')
                print("🔴 Detected: Both Pinched → Start Presentation (F5)")
                sys.stdout.flush()
                action_taken = True
            
            # Gesture 4: Both Open → End Presentation (ESC)
            elif left_open and right_open:
                pyautogui.press('esc')
                print("🔴 Detected: Both Open → End Presentation (ESC)")
                sys.stdout.flush()
                action_taken = True
            
            if action_taken:
                self.last_gesture_time = current_time
        
        return action_taken
    
    def run(self):
        """Main loop: capture video, process gestures, and show live camera preview."""
        cap, camera_index, backend_name = self._open_camera()
        
        if cap is None:
            print("❌ Error: Could not open a usable webcam feed")
            print("   Try closing other camera apps (Zoom/Teams/Photo Booth) and allow Camera permission.")
            return
        
        print("\n✅ Two-Hand PowerPoint Controller Started")
        print("━" * 50)
        print(f"📹 Webcam: Ready (index {camera_index}, backend {backend_name})")
        print("\n🎯 Gesture Mappings:")
        print("  • Left Fist + Right Point  → Previous Slide")
        print("  • Right Fist + Left Point  → Next Slide")
        print("  • Both Pinched             → Start Presentation (F5)")
        print("  • Both Open                → End Presentation (ESC)")
        print("\n💡 Tip: Open PowerPoint before showing gestures!")
        print("⏹️  Press 'q' in camera window or Ctrl+C to quit\n")
        print("━" * 50)
        print("Waiting for gestures...\n")
        
        frame_count = 0
        black_frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Failed to read frame")
                    break

                if self._is_frame_black(frame):
                    black_frame_count += 1
                else:
                    black_frame_count = 0
                
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Process frame for gestures
                self.process_frame(frame)

                cv2.putText(
                    frame,
                    "Two-Hand Controller: press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if black_frame_count > 15:
                    cv2.putText(
                        frame,
                        "Camera feed is dark/black. Check camera permission and lighting.",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow('Two-Hand PowerPoint Controller', frame)
                
                frame_count += 1
                
                # Print status every 30 frames (roughly every second at ~30fps)
                if frame_count % 30 == 0:
                    print("  ✓ Still listening for gestures...", end='\r')
                    sys.stdout.flush()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n⏹️  Shutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("✅ Controller stopped")

if __name__ == '__main__':
    controller = TwoHandPowerPointController()
    controller.run()

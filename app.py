import os
os.environ['GLOG_minloglevel'] = '2'  # Suppresses INFO and WARNING messages

import cv2
import mediapipe as mp
from controller import Controller  # Assuming Controller is in controller.py
import pyautogui
from collections import deque
import numpy as np

# Constants for configuration
CAM_WIDTH = 640
CAM_HEIGHT = 480
FRAME_REDUCTION = 100
SMOOTHING_FACTOR = 20  # Increased for smoother movement
POS_HISTORY_LENGTH = 10  # Longer history for more stable averaging
TARGET_FPS = 60  # Higher FPS for smoother cursor updates
FRAME_TIME = 1 / TARGET_FPS

# Initialize video capture and hand tracking
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Create Controller instance
controller = Controller(CAM_WIDTH, CAM_HEIGHT, FRAME_REDUCTION)

# Enhanced smoothing parameters
pos_history = deque(maxlen=POS_HISTORY_LENGTH)
prev_x, prev_y = 0, 0

def smooth_coordinates(x: float, y: float) -> tuple[float, float]:
    """Apply enhanced exponential moving average smoothing"""
    global prev_x, prev_y
    alpha = 0.15  # Lower alpha for smoother, less responsive movement (adjustable: 0.1-0.3)
    smooth_x = alpha * x + (1 - alpha) * prev_x
    smooth_y = alpha * y + (1 - alpha) * prev_y
    prev_x, prev_y = smooth_x, smooth_y
    return smooth_x, smooth_y

def get_position_with_frame_reduction(x, y, frame_reduction, img_shape):
    """Convert coordinates to screen coordinates with frame reduction. 
    Frame reduction is used to prevent the cursor from reaching the edges of the screen."""
    h, w, _ = img_shape
    x = np.interp(x, (frame_reduction, w - frame_reduction), (0, controller.screen_width))
    y = np.interp(y, (frame_reduction, h - frame_reduction), (0, controller.screen_height))
    return x, y

while True:
    start_time = cv2.getTickCount()
    
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        controller.hand_landmarks = results.multi_hand_landmarks[0]
        controller.img_shape = img.shape
        
        # Draw landmarks
        mpDraw.draw_landmarks(img, controller.hand_landmarks, mpHands.HAND_CONNECTIONS)
        
        # Label landmarks (optional - comment out for final version)
        for id, lm in enumerate(controller.hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Update finger states
        controller.update_fingers_status()

        # Display finger status on the screen
        finger_status_text = [
            f"Little Finger: {'Down' if controller.little_finger_down else 'Up'}",
            f"Ring Finger: {'Down' if controller.ring_finger_down else 'Up'}",
            f"Middle Finger: {'Down' if controller.middle_finger_down else 'Up'}",
            f"Index Finger: {'Down' if controller.index_finger_down else 'Up'}",
            f"Thumb Finger: {'Down' if controller.thumb_finger_down else 'Up'}"
        ]
        for i, text in enumerate(finger_status_text):
            cv2.putText(img, text, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Get index finger tip position
        x1 = controller.hand_landmarks.landmark[8].x
        y1 = controller.hand_landmarks.landmark[8].y
        h, w, c = img.shape
        x1, y1 = int(x1 * w), int(y1 * h)

        # Apply enhanced smoothing
        smooth_x, smooth_y = smooth_coordinates(x1, y1)
        pos_history.append((smooth_x, smooth_y))
        
        # Use weighted average of recent positions for extra smoothness
        if len(pos_history) == POS_HISTORY_LENGTH:
            weights = np.linspace(0.5, 1.0, POS_HISTORY_LENGTH)  # Newer positions weighted more
            avg_x = np.average([pos[0] for pos in pos_history], weights=weights)
            avg_y = np.average([pos[1] for pos in pos_history], weights=weights)
            
            # Convert to screen coordinates with frame reduction
            avg_x, avg_y = get_position_with_frame_reduction(avg_x, avg_y, FRAME_REDUCTION, img.shape)
            
            # Update cursor position with smoothed coordinates
            controller.cursor_moving(avg_x, avg_y)
            
            # Additional control functions
            controller.detect_clicking()
            controller.detect_dragging()
            controller.detect_scrolling()
            controller.detect_zooming()
            
            # Update and display gesture description
            controller.update_gesture_description()
            cv2.putText(img, controller.gesture_description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    # cv2.imshow('Hand Tracker', img)
    
    # Frame rate control
    end_time = cv2.getTickCount()
    elapsed = (end_time - start_time) / cv2.getTickFrequency()
    sleep_time = max(0, FRAME_TIME - elapsed)
    if cv2.waitKey(max(1, int(sleep_time * 1000))) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

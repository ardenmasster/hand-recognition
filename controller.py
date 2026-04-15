import pyautogui
import time
import numpy as np

class Controller:
    def __init__(self, cam_width, cam_height, frame_reduction):
        self.prev_hand = None
        self.right_clicked = False
        self.left_clicked = False
        self.double_clicked = False
        self.dragging = False
        self.hand_landmarks = None
        self.little_finger_down = None
        self.little_finger_up = None
        self.index_finger_down = None
        self.index_finger_up = None
        self.middle_finger_down = None
        self.middle_finger_up = None
        self.ring_finger_down = None
        self.ring_finger_up = None
        self.thumb_finger_down = None  # Fixed typo
        self.thumb_finger_up = None    # Fixed typo
        self.all_fingers_down = None
        self.all_fingers_up = None
        self.index_finger_within_thumb = None
        self.middle_finger_within_thumb = None
        self.little_finger_within_thumb = None
        self.ring_finger_within_thumb = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.gesture_description = "Tracking"  # Instance attribute
        self.smoothening = 7  # Add smoothening factor
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.frame_reduction = frame_reduction

    def update_fingers_status(self):
        if not self.hand_landmarks:
            return
        
        # Finger status updates
        self.little_finger_down = self.hand_landmarks.landmark[20].y > self.hand_landmarks.landmark[17].y
        self.little_finger_up = self.hand_landmarks.landmark[20].y < self.hand_landmarks.landmark[17].y
        self.index_finger_down = self.hand_landmarks.landmark[8].y > self.hand_landmarks.landmark[5].y
        self.index_finger_up = self.hand_landmarks.landmark[8].y < self.hand_landmarks.landmark[5].y
        self.middle_finger_down = self.hand_landmarks.landmark[12].y > self.hand_landmarks.landmark[9].y
        self.middle_finger_up = self.hand_landmarks.landmark[12].y < self.hand_landmarks.landmark[9].y
        self.ring_finger_down = self.hand_landmarks.landmark[16].y > self.hand_landmarks.landmark[13].y
        self.ring_finger_up = self.hand_landmarks.landmark[16].y < self.hand_landmarks.landmark[13].y
        self.thumb_finger_down = self.hand_landmarks.landmark[4].y > self.hand_landmarks.landmark[13].y
        self.thumb_finger_up = self.hand_landmarks.landmark[4].y < self.hand_landmarks.landmark[13].y
        
        # Combined states
        self.all_fingers_down = (self.index_finger_down and self.middle_finger_down and 
                               self.ring_finger_down and self.little_finger_down)
        self.all_fingers_up = (self.index_finger_up and self.middle_finger_up and 
                             self.ring_finger_up and self.little_finger_up)
        
        # Finger-thumb interactions
        self.index_finger_within_thumb = (self.hand_landmarks.landmark[8].y > self.hand_landmarks.landmark[4].y and 
                                        self.hand_landmarks.landmark[8].y < self.hand_landmarks.landmark[2].y)
        self.middle_finger_within_thumb = (self.hand_landmarks.landmark[12].y > self.hand_landmarks.landmark[4].y and 
                                         self.hand_landmarks.landmark[12].y < self.hand_landmarks.landmark[2].y)
        self.little_finger_within_thumb = (self.hand_landmarks.landmark[20].y > self.hand_landmarks.landmark[4].y and 
                                         self.hand_landmarks.landmark[20].y < self.hand_landmarks.landmark[2].y)
        self.ring_finger_within_thumb = (self.hand_landmarks.landmark[16].y > self.hand_landmarks.landmark[4].y and 
                                       self.hand_landmarks.landmark[16].y < self.hand_landmarks.landmark[2].y)

    def get_position(self, hand_result):
            """
            It tracks your hand, maps it to the screen, dampens small jitters, smooths motion, and keeps the cursor on-screen
            
            Combines distance-based damping with low-pass filtering for smooth, responsive motion.
            
            Args:
                hand_result: Hand tracking result with landmark data.
            
            Returns:
                tuple(int, int): Smoothed (x, y) cursor coordinates.
            """
            # Extract hand position (using landmark 9, like first function)
            point = 9
            hand_x = hand_result.landmark[point].x  # Normalized 0-1
            hand_y = hand_result.landmark[point].y  # Normalized 0-1

            # Map to screen coordinates with frame reduction (like second function)
            x_mapped = np.interp(hand_x,
                                (self.frame_reduction / self.cam_width, 1 - self.frame_reduction / self.cam_width),
                                (0, self.screen_width))
            y_mapped = np.interp(hand_y,
                                (self.frame_reduction / self.cam_height, 1 - self.frame_reduction / self.cam_height),
                                (0, self.screen_height))

            # Initialize previous hand position if None
            if self.prev_hand_x is None or self.prev_hand_y is None:
                self.prev_hand_x, self.prev_hand_y = x_mapped, y_mapped

            # Calculate movement delta (like first function)
            delta_x = x_mapped - self.prev_hand_x
            delta_y = y_mapped - self.prev_hand_y
            distsq = delta_x**2 + delta_y**2

            # Distance-based ratio (inspired by first function, tuned for smoother feel)
            if distsq <= 25:  # Small movements: heavy damping
                ratio = 0
            elif distsq <= 900:  # Medium movements: proportional scaling
                ratio = 0.05 * (distsq ** 0.5)  # Reduced from 0.07 for smoother transitions
            else:  # Large movements: cap for control
                ratio = 1.5  # Reduced from 2.1 to avoid overshooting

            # Apply low-pass filter (like second function) with distance-based adjustment
            curr_x = self.prev_x + (delta_x * ratio) / self.smoothening
            curr_y = self.prev_y + (delta_y * ratio) / self.smoothening

            # Update previous positions
            self.prev_x, self.prev_y = curr_x, curr_y
            self.prev_hand_x, self.prev_hand_y = x_mapped, y_mapped

            # Ensure within screen bounds (like second function)
            curr_x = max(0, min(curr_x, self.screen_width))
            curr_y = max(0, min(curr_y, self.screen_height))

            return int(curr_x), int(curr_y)

    def cursor_moving(self, x=None, y=None):
        if not self.hand_landmarks:
            return
            
        if x is None or y is None:  # Use hand tracking if no coordinates provided
            point = 9  # Middle finger MCP
            current_x, current_y = self.hand_landmarks.landmark[point].x, self.hand_landmarks.landmark[point].y
            x, y = self.get_position(current_x, current_y)
        
        cursor_freezed = self.all_fingers_up and self.thumb_finger_down
        if not cursor_freezed:
            pyautogui.moveTo(x, y, duration=0)

    def detect_scrolling(self):
        scroll_step = 20  # Smaller steps for bit-by-bit scrolling (adjustable)
        scroll_cooldown = 0.5  # Time in seconds between scroll steps
        current_time = time.time()
        
        # Check if enough time has passed since last scroll
        if not hasattr(self, '_last_scroll_time'):
            self._last_scroll_time = 0
        
        if current_time - self._last_scroll_time < scroll_cooldown:
            return

        # Scrolling up: middle, ring, and little fingers down, index finger up
        scroll_up_condition = (self.middle_finger_down and self.ring_finger_down and 
                               self.little_finger_down and self.index_finger_up)

        # Scrolling down: index, middle, and ring fingers down, little finger up
        scroll_down_condition = (self.index_finger_down and self.middle_finger_down and 
                                 self.ring_finger_down and self.little_finger_up)

        if scroll_up_condition:
            pyautogui.scroll(scroll_step)  # Scroll up
            pyautogui.scroll(scroll_step)  # Scroll up
            self.gesture_description = "Scrolling UP"
            self._last_scroll_time = current_time

        elif scroll_down_condition:
            pyautogui.scroll(-scroll_step)  # Scroll down
            pyautogui.scroll(-scroll_step)  # Scroll down
            self.gesture_description = "Scrolling DOWN"
            self._last_scroll_time = current_time

    def detect_zooming(self):
        # Zoom in: index and middle fingers up, ring and little fingers down
        zooming_in = (self.index_finger_up and self.middle_finger_up and 
                     self.ring_finger_down and self.little_finger_down)
        
        # Zoom out: index and little fingers up, middle and ring fingers down
        zooming_out = (self.index_finger_up and self.little_finger_up and 
                      self.middle_finger_down and self.ring_finger_down)
        
        if zooming_out:
            pyautogui.keyDown('ctrl')
            pyautogui.press('-')
            pyautogui.keyUp('ctrl')
            self.gesture_description = "Zooming Out"

        if zooming_in:
            pyautogui.keyDown('ctrl')
            pyautogui.press('+')
            pyautogui.keyUp('ctrl')
            self.gesture_description = "Zooming In"

    def detect_clicking(self):
        left_click_condition = (self.index_finger_within_thumb and self.middle_finger_up and 
                              self.ring_finger_up and self.little_finger_up and 
                              not self.middle_finger_within_thumb and not self.ring_finger_within_thumb and 
                              not self.little_finger_within_thumb)
        if not self.left_clicked and left_click_condition:
            pyautogui.click()
            self.left_clicked = True
            self.gesture_description = "Left Clicking"
        elif not self.index_finger_within_thumb:
            self.left_clicked = False

        right_click_condition = (self.middle_finger_within_thumb and self.index_finger_up and 
                               self.ring_finger_up and self.little_finger_up and 
                               not self.index_finger_within_thumb and not self.ring_finger_within_thumb and 
                               not self.little_finger_within_thumb)
        if not self.right_clicked and right_click_condition:
            pyautogui.rightClick()
            self.right_clicked = True
            self.gesture_description = "Right Clicking"
        elif not self.middle_finger_within_thumb:
            self.right_clicked = False

        double_click_condition = (self.ring_finger_within_thumb and self.index_finger_up and 
                                self.middle_finger_up and self.little_finger_up and 
                                not self.index_finger_within_thumb and not self.middle_finger_within_thumb and 
                                not self.little_finger_within_thumb)
        if not self.double_clicked and double_click_condition:
            pyautogui.doubleClick()
            self.double_clicked = True
            self.gesture_description = "Double Clicking"
        elif not self.ring_finger_within_thumb:
            self.double_clicked = False

    def detect_dragging(self):
        if not self.dragging and self.all_fingers_down:
            pyautogui.mouseDown(button="left")
            self.dragging = True
            self.gesture_description = "Dragging"
        elif not self.all_fingers_down and self.dragging:
            pyautogui.mouseUp(button="left")
            self.dragging = False

    def update_gesture_description(self):
        """Update gesture description based on current state and actions"""
        if self.dragging:
            self.gesture_description = "Dragging"
        elif self.left_clicked:
            self.gesture_description = "Left Click"
        elif self.right_clicked:
            self.gesture_description = "Right Click"
        elif self.double_clicked:
            self.gesture_description = "Double Click"
        elif self.all_fingers_up and self.thumb_finger_down:
            self.gesture_description = "Cursor Frozen"
        elif self.all_fingers_down:
            self.gesture_description = "Ready to Drag"
        elif self.index_finger_up and self.middle_finger_up and self.ring_finger_down and self.little_finger_down:
            self.gesture_description = "Ready to Zoom In"
        elif self.index_finger_up and self.little_finger_up and self.middle_finger_down and self.ring_finger_down:
            self.gesture_description = "Ready to Zoom Out"
        elif self.middle_finger_down and self.ring_finger_down and self.little_finger_down and self.index_finger_up:
            self.gesture_description = "Scrolling UP"
        elif self.index_finger_down and self.middle_finger_down and self.ring_finger_down and self.little_finger_up:
            self.gesture_description = "Scrolling DOWN"
        else:
            self.gesture_description = "Tracking"

    def print_finger_status(self):
        """Print the status of each finger (up or down)"""
        print(f"Little Finger: {'Down' if self.little_finger_down else 'Up'}")
        print(f"Ring Finger: {'Down' if self.ring_finger_down else 'Up'}")
        print(f"Middle Finger: {'Down' if self.middle_finger_down else 'Up'}")
        print(f"Index Finger: {'Down' if self.index_finger_down else 'Up'}")
        print(f"Thumb Finger: {'Down' if self.thumb_finger_down else 'Up'}")


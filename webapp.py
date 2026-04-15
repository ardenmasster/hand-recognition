import atexit
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request


@dataclass
class TrackerConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    gesture_hold_frames: int = 4
    action_cooldown_ms: int = 450
    smoothing_alpha: float = 0.18


class StableHandTracker:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = TrackerConfig()

        self.cap = None
        self.hands = None
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.frame_width = 960
        self.frame_height = 540
        self.camera_index = int(os.environ.get("CAMERA_INDEX", "-1"))
        self.active_camera_index = None
        self.last_frame_brightness = 0.0

        self.prev_time = time.time()
        self.fps_samples = deque(maxlen=30)

        self.last_gesture = "No Hand"
        self.raw_candidate = "No Hand"
        self.candidate_frames = 0
        self.last_action_time = 0.0

        self.total_frames = 0
        self.hands_detected_frames = 0
        self.gesture_counts = Counter()
        self.event_log = deque(maxlen=10)
        self.started_at = time.time()

    def _open_camera_candidate(self, index):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            cap.release()
            return None, -1.0

        brightness_samples = []
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_samples.append(float(np.mean(gray)))

        if not brightness_samples:
            cap.release()
            return None, -1.0

        return cap, float(np.mean(brightness_samples))

    def _ensure_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return

        preferred_indices = [self.camera_index] if self.camera_index >= 0 else [0, 1, 2]
        best = (None, None, -1.0)

        for index in preferred_indices:
            candidate, brightness = self._open_camera_candidate(index)
            if candidate is None:
                continue

            # Prefer sources that are not near-black.
            if brightness > best[2]:
                if best[0] is not None:
                    best[0].release()
                best = (candidate, index, brightness)
            else:
                candidate.release()

        if best[0] is not None:
            self.cap = best[0]
            self.active_camera_index = best[1]
            self.last_frame_brightness = best[2]
        else:
            self.cap = None
            self.active_camera_index = None

    def _ensure_hands(self):
        if self.hands is not None:
            return
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def _rebuild_hands(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        self._ensure_hands()

    @staticmethod
    def _distance(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return (dx * dx + dy * dy) ** 0.5

    def _finger_states(self, hand_landmarks):
        lm = hand_landmarks.landmark
        states = {
            "index_up": lm[8].y < lm[5].y,
            "middle_up": lm[12].y < lm[9].y,
            "ring_up": lm[16].y < lm[13].y,
            "little_up": lm[20].y < lm[17].y,
            "thumb_up": lm[4].y < lm[3].y,
        }

        palm_scale = max(0.001, self._distance(lm[0], lm[9]))
        index_pinch = self._distance(lm[8], lm[4]) / palm_scale < 0.35
        middle_pinch = self._distance(lm[12], lm[4]) / palm_scale < 0.35
        ring_pinch = self._distance(lm[16], lm[4]) / palm_scale < 0.35

        states["index_pinch"] = index_pinch
        states["middle_pinch"] = middle_pinch
        states["ring_pinch"] = ring_pinch
        return states

    def _classify_gesture(self, states):
        index_up = states["index_up"]
        middle_up = states["middle_up"]
        ring_up = states["ring_up"]
        little_up = states["little_up"]

        all_up = index_up and middle_up and ring_up and little_up
        all_down = (not index_up) and (not middle_up) and (not ring_up) and (not little_up)

        if states["index_pinch"] and middle_up and ring_up and little_up:
            return "Left Click"
        if states["middle_pinch"] and index_up and ring_up and little_up:
            return "Right Click"
        if states["ring_pinch"] and index_up and middle_up and little_up:
            return "Double Click"
        if middle_up and ring_up and (not index_up) and (not little_up):
            return "Scroll Down"
        if index_up and (not middle_up) and (not ring_up) and (not little_up):
            return "Scroll Up"
        if index_up and middle_up and (not ring_up) and (not little_up):
            return "Zoom In"
        if index_up and little_up and (not middle_up) and (not ring_up):
            return "Zoom Out"
        if all_down:
            return "Drag"
        if all_up:
            return "Open Palm"
        return "Tracking"

    def _stabilize_gesture(self, raw_gesture):
        now = time.time()

        if raw_gesture == self.raw_candidate:
            self.candidate_frames += 1
        else:
            self.raw_candidate = raw_gesture
            self.candidate_frames = 1

        if self.candidate_frames < self.config.gesture_hold_frames:
            return self.last_gesture

        action_gestures = {
            "Left Click",
            "Right Click",
            "Double Click",
            "Scroll Up",
            "Scroll Down",
            "Zoom In",
            "Zoom Out",
        }

        if raw_gesture in action_gestures:
            cooldown_s = self.config.action_cooldown_ms / 1000.0
            if now - self.last_action_time < cooldown_s:
                return self.last_gesture
            self.last_action_time = now

        if raw_gesture != self.last_gesture:
            self.last_gesture = raw_gesture
            self.gesture_counts[raw_gesture] += 1
            self.event_log.appendleft(
                {
                    "gesture": raw_gesture,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )

        return self.last_gesture

    def _draw_hud(self, frame, gesture, hand_found, fps):
        status_color = (40, 210, 120) if hand_found else (45, 130, 255)

        cv2.rectangle(frame, (0, 0), (470, 120), (20, 20, 20), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(
            frame,
            f"Detection: {self.config.min_detection_confidence:.2f}  Tracking: {self.config.min_tracking_confidence:.2f}",
            (16, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

    def _placeholder_frame(self, text):
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else b""

    def get_frame_jpeg(self):
        with self.lock:
            self._ensure_camera()
            self._ensure_hands()

            if not self.cap or not self.cap.isOpened():
                return self._placeholder_frame("Camera unavailable. Check permission settings.")

            success, frame = self.cap.read()
            if not success or frame is None:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                return self._placeholder_frame("Failed to read camera frame.")

            self.total_frames += 1
            self.last_frame_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            hand_found = bool(results.multi_hand_landmarks)
            current_gesture = self.last_gesture

            if hand_found:
                self.hands_detected_frames += 1
                hand = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                states = self._finger_states(hand)
                raw_gesture = self._classify_gesture(states)
                current_gesture = self._stabilize_gesture(raw_gesture)
            else:
                current_gesture = self._stabilize_gesture("No Hand")

            now = time.time()
            dt = max(1e-5, now - self.prev_time)
            self.prev_time = now
            fps = 1.0 / dt
            self.fps_samples.append(fps)
            avg_fps = float(np.mean(self.fps_samples)) if self.fps_samples else 0.0

            self._draw_hud(frame, current_gesture, hand_found, avg_fps)

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return self._placeholder_frame("Failed to encode camera frame.")
            return buffer.tobytes()

    def get_status(self):
        with self.lock:
            fps = float(np.mean(self.fps_samples)) if self.fps_samples else 0.0
            uptime = int(time.time() - self.started_at)
            detection_rate = (
                round((self.hands_detected_frames / self.total_frames) * 100.0, 1)
                if self.total_frames > 0
                else 0.0
            )
            return {
                "gesture": self.last_gesture,
                "fps": round(fps, 1),
                "uptime_seconds": uptime,
                "total_frames": self.total_frames,
                "hand_detection_rate": detection_rate,
                "camera_index": self.active_camera_index,
                "frame_brightness": round(self.last_frame_brightness, 1),
                "config": {
                    "min_detection_confidence": self.config.min_detection_confidence,
                    "min_tracking_confidence": self.config.min_tracking_confidence,
                    "gesture_hold_frames": self.config.gesture_hold_frames,
                    "action_cooldown_ms": self.config.action_cooldown_ms,
                    "smoothing_alpha": self.config.smoothing_alpha,
                },
                "gesture_counts": dict(self.gesture_counts),
                "recent_events": list(self.event_log),
            }

    def update_config(self, payload):
        with self.lock:
            old_detection = self.config.min_detection_confidence
            old_tracking = self.config.min_tracking_confidence

            detection = payload.get("min_detection_confidence", self.config.min_detection_confidence)
            tracking = payload.get("min_tracking_confidence", self.config.min_tracking_confidence)
            hold_frames = payload.get("gesture_hold_frames", self.config.gesture_hold_frames)
            cooldown_ms = payload.get("action_cooldown_ms", self.config.action_cooldown_ms)
            smoothing_alpha = payload.get("smoothing_alpha", self.config.smoothing_alpha)

            self.config.min_detection_confidence = float(np.clip(detection, 0.3, 0.95))
            self.config.min_tracking_confidence = float(np.clip(tracking, 0.3, 0.95))
            self.config.gesture_hold_frames = int(np.clip(hold_frames, 2, 12))
            self.config.action_cooldown_ms = int(np.clip(cooldown_ms, 100, 2000))
            self.config.smoothing_alpha = float(np.clip(smoothing_alpha, 0.05, 0.8))

            if (
                abs(old_detection - self.config.min_detection_confidence) > 1e-9
                or abs(old_tracking - self.config.min_tracking_confidence) > 1e-9
            ):
                self._rebuild_hands()
            return self.get_status()

    def reset_stats(self):
        with self.lock:
            self.total_frames = 0
            self.hands_detected_frames = 0
            self.gesture_counts = Counter()
            self.event_log = deque(maxlen=10)
            self.started_at = time.time()

    def close(self):
        with self.lock:
            if self.hands is not None:
                self.hands.close()
                self.hands = None
            if self.cap is not None:
                self.cap.release()
                self.cap = None


app = Flask(__name__)
tracker = StableHandTracker()


def mjpeg_generator():
    while True:
        frame = tracker.get_frame_jpeg()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def status():
    return jsonify(tracker.get_status())


@app.route("/api/config", methods=["POST"])
def update_config():
    payload = request.get_json(silent=True) or {}
    return jsonify(tracker.update_config(payload))


@app.route("/api/reset", methods=["POST"])
def reset_stats():
    tracker.reset_stats()
    return jsonify({"ok": True})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@atexit.register
def _cleanup():
    tracker.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

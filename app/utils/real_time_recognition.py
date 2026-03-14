import cv2
import numpy as np
import mediapipe as mp
import json
import joblib
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import threading
import time
import os
import platform

os.environ['TF_USE_LEGACY_KERAS'] = '1'
warnings.filterwarnings("ignore")


class RealTimeSignLanguageRecognizer:
    def __init__(
            self,
            model_path="app/model/sign_language_recognition_fixed.keras",
            scaler_path="app/model/scaler.pkl",
            label_encoder_path="app/model/label_encoder.pkl",
            feature_order_path="app/model/feature_order.json",
            cam_id=None,
    ):
        print("Initializing RealTimeSignLanguageRecognizer with 297 FEATURES...")

        # Модель файлын тексеру
        self.model_available = os.path.exists(model_path)
        self.prediction_history = deque(maxlen=5)

        if self.model_available:
            try:
                self.model = load_model(model_path)
                print(f"✓ Model loaded: {model_path}")
                # Модель input shape тексеру
                if hasattr(self.model, 'input_shape'):
                    print(f"✓ Model input shape: {self.model.input_shape}")
            except Exception as e:
                print(f"✗ Model load error: {e}")
                self.model = None
                self.model_available = False
        else:
            print(f"⚠ Model file not found: {model_path}")
            self.model = None

        # Scaler жүктеу (99 features күтуі керек)
        try:
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            if self.scaler:
                print(f"✓ Scaler loaded: expects {self.scaler.n_features_in_} features")
                if self.scaler.n_features_in_ != 297:
                    print(f"⚠ WARNING: Scaler expects {self.scaler.n_features_in_}, but we use 297 features")
        except Exception as e:
            print(f"✗ Scaler load error: {e}")
            self.scaler = None

        # Label encoder жүктеу
        try:
            self.label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None
            if self.label_encoder:
                print(f"✓ Label encoder loaded: {len(self.label_encoder.classes_)} classes")
        except Exception as e:
            print(f"✗ Label encoder load error: {e}")
            self.label_encoder = None

        # Feature order жүктеу
        if os.path.exists(feature_order_path):
            try:
                with open(feature_order_path, "r") as f:
                    self.feature_order = json.load(f)
                print(f"📊 Loaded feature order with {len(self.feature_order)} features")
            except Exception as e:
                print(f"✗ Feature order load error: {e}")
                self.feature_order = [f"f_{i}" for i in range(297)]
        else:
            self.feature_order = [f"f_{i}" for i in range(297)]
            print(f"📊 Created default feature order with 198 features")

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        self.pose = None

        # # Parameters for 99 FEATURES
        # self.fps = 15
        # self.window_sec = 0.5  # 0.5 секунд = 7-8 кадр
        # self.window_size = int(self.fps * self.window_sec)
        # self.sequence_length = 5  # 5 windows = ~2.5 секунд видео

        self.fps = 15
        self.window_sec = 0.2

        self.window_size = 5
        self.sequence_length = 5



        # Камера индексін таңдау
        if cam_id is None:
            self.CAM_ID = self.find_working_camera()
        else:
            self.CAM_ID = cam_id

        # Buffers
        self.frame_buffer = deque(maxlen=self.window_size)
        self.feature_buffer = deque(maxlen=self.sequence_length)

        # Landmark indices for 99 FEATURES
        # ALL pose landmarks (33 нүкте)
        self.pose_landmarks = list(range(33))  # 0-32

        # Hand landmarks (21 нүкте) - бірақ бізге тек координаттар керек
        self.hand_landmarks = list(range(21))  # 0-20

        # For FPS calculation
        self.frame_count = 0
        self.fps_timer = cv2.getTickCount()

        # Video capture
        self.cap = None
        self.is_running = False

        # Current predictions
        self.current_prediction = None
        self.top3_predictions = None
        self.lock = threading.Lock()

        print(f"✓ Initialization complete! (Model available: {self.model_available})")
        print(f"📊 Configuration: window_size={self.window_size}, sequence_length={self.sequence_length}")

    # ===== ЖҰМЫС ІСТЕЙТІН КАМЕРАНЫ ТАБУ =====
    def find_working_camera(self, max_check=5):
        print("\n🔍 Looking for working cameras...")
        working_cameras = []

        for i in range(max_check):
            try:
                print(f"  Testing camera index {i}...", end=" ")
                cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ WORKING (frame size: {frame.shape})")
                        working_cameras.append(i)
                    else:
                        print(f"✗ Can't read frame")

                    cap.release()
                else:
                    print(f"✗ Not available")

            except Exception as e:
                print(f"✗ Error: {e}")

        if working_cameras:
            print(f"\n✅ Found working cameras: {working_cameras}")
            print(f"👉 Using camera index: {working_cameras[0]}")
            return working_cameras[0]

        print("\n⚠️ No working cameras found, using default index 0")
        return 0

    def _init_mediapipe(self):
        """Initialize MediaPipe models"""
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.35,
               min_tracking_confidence = 0.35
            )



        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.35,
                min_tracking_confidence = 0.35
            )

    def _cleanup_mediapipe(self):
        """Clean up MediaPipe models"""
        if self.hands:
            self.hands.close()
            self.hands = None
        if self.pose:
            self.pose.close()
            self.pose = None

    def start_capture(self):
        """Start video capture"""
        try:
            if self.cap is None or not self.cap.isOpened():
                print(f"Attempting to open camera with ID: {self.CAM_ID}")
                self.cap = cv2.VideoCapture(self.CAM_ID)

                if not self.cap.isOpened():
                    print(f"Failed to open camera {self.CAM_ID}, searching for alternatives...")
                    for cam_idx in [0, 1, 2, 3, 4]:
                        if cam_idx == self.CAM_ID:
                            continue
                        print(f"Trying camera index {cam_idx}...")
                        self.cap = cv2.VideoCapture(cam_idx)
                        if self.cap.isOpened():
                            print(f"✓ Camera opened successfully at index {cam_idx}")
                            self.CAM_ID = cam_idx
                            break
                    else:
                        print("✗ Failed to open any camera")
                        return False

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            self._init_mediapipe()
            self.frame_buffer.clear()
            self.feature_buffer.clear()
            self.prediction_history.clear()
            self.is_running = True
            print(f"✓ Camera {self.CAM_ID} started successfully")
            return True

        except Exception as e:
            print(f"✗ Error starting capture: {e}")
            return False

    def stop_capture(self):
        """Stop video capture and release resources"""
        self.is_running = False
        self._cleanup_mediapipe()
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.current_prediction = None
            self.top3_predictions = None

    def extract_landmarks_from_frame(self, frame):
        """Extract ALL hand and pose landmarks from a single frame for 99 features"""
        if self.hands is None or self.pose is None:
            return {"hand_0": [], "hand_1": [], "pose": []}, None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        landmarks = {
            "hand_0": [],
            "hand_1": [],
            "hand_labels": [],
            "pose": [],
        }

        # === БАРЛЫҚ HAND LANDMARKS АЛУ (21 нүкте) ===
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if hand_idx > 1:
                    break

                label = hands_results.multi_handedness[hand_idx].classification[0].label
                landmarks["hand_labels"].append(label)

                for lm_idx in self.hand_landmarks:
                    lm = hand_landmarks.landmark[lm_idx]
                    landmarks[f"hand_{hand_idx}"].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": 1.0
                    })

        # === БАРЛЫҚ POSE LANDMARKS АЛУ (33 нүкте) ===
        if pose_results.pose_landmarks:
            for idx in self.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark[idx]
                landmarks["pose"].append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 1.0
                })

        return landmarks, hands_results, pose_results


    def predict(self):
        """Make prediction with 297 features"""
        if not self.model_available or self.model is None:
            return None, None

        if len(self.feature_buffer) < self.sequence_length:
            return None, None

        try:
            X = np.array(list(self.feature_buffer))
            print(f"📊 Feature buffer shape: {X.shape}")

            X = X.reshape(1, self.sequence_length, 297)
            print(f"📊 Reshaped for model: {X.shape}")

            if self.scaler:
                X_reshaped = X.reshape(-1, 297)
                print(f"📊 Scaling features...")

                X_scaled = self.scaler.transform(X_reshaped)
                X = X_scaled.reshape(1, self.sequence_length, 297)
                print(f"📊 After scaling: {X.shape}")
            else:
                print("⚠️ No scaler available, using raw features")

            predictions = self.model.predict(X, verbose=0)[0]
            print(f"✅ Prediction shape: {predictions.shape}")

            top3_indices = np.argsort(predictions)[-3:][::-1]

            if self.label_encoder:
                top3_labels = self.label_encoder.inverse_transform(top3_indices)
            else:
                top3_labels = [f"class_{i}" for i in top3_indices]

            top3_confidences = predictions[top3_indices]

            # confidence threshold
            if top3_confidences[0] < 0.12:
                predicted_label = None
            else:
                predicted_label = top3_labels[0]

                self.prediction_history.append(predicted_label)

                if len(self.prediction_history) == 5:
                    predicted_label = max(
                        set(self.prediction_history),
                        key=self.prediction_history.count
                    )

            latest = self.feature_buffer[-1]

            for i in range(min(20, len(self.feature_order))):
                print(self.feature_order[i], latest[i])

            if predicted_label:
                print(f"🎯 Predicted: {predicted_label} ({top3_confidences[0] * 100:.1f}%)")
            else:
                print(f"⚠ Low confidence: {top3_confidences[0] * 100:.1f}%")

            return predicted_label, list(zip(top3_labels, top3_confidences * 100))

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def extract_window_features(self, frames):
        """
        frames: 5 landmark windows
        output: 297 features
        33 landmarks × 3 axis × (current + std + delta)
        """

        result = []

        for i in range(33):

            for axis in ['x', 'y', 'z']:
                vals = []

                for frame in frames:

                    if i < len(frame["pose"]):
                        landmark = frame["pose"][i]

                        if axis in landmark:
                            value = landmark[axis]

                            # z clamp
                            if axis == 'z':
                                value = max(min(value, 1.0), -1.0)

                            vals.append(value)
                        else:
                            vals.append(0.0)
                    else:
                        vals.append(0.0)

                current_val = vals[-1] * 0.5

                std_val = min(np.std(vals) * 3, 0.2)

                if len(vals) >= 2:
                    delta = (vals[-1] - vals[-2]) * 5
                    delta = max(min(delta, 0.2), -0.2)
                else:
                    delta = 0.0

                if axis == 'z':
                    current_val *= 0.1
                    std_val *= 0.1
                    delta *= 0.1

                result.append(current_val)
                result.append(std_val)
                result.append(delta)

        return np.array(result)


    def draw_landmarks(self, frame, hands_results, pose_results):
        """Draw MediaPipe landmarks on frame"""
        # Draw hand landmarks
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

        # Draw ALL pose landmarks
        if pose_results and pose_results.pose_landmarks:
            # Draw all 33 landmarks
            for idx in self.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark[idx]
                h, w = frame.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Different colors for different body parts
                if idx in [11, 12]:  # Shoulders
                    color = (255, 0, 0)  # Blue
                elif idx in [13, 14]:  # Elbows
                    color = (0, 255, 0)  # Green
                elif idx in [15, 16]:  # Wrists
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Yellow

                cv2.circle(frame, (cx, cy), 3, color, -1)

        return frame

    def get_frame(self):
        """Get a single processed frame for streaming only"""
        if not self.is_running:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera Stopped",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            return placeholder

        if not self.cap or not self.cap.isOpened():
            if not self.start_capture():
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    error_frame,
                    "Camera Error",
                    (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                return error_frame

        ret, frame = self.cap.read()
        if not ret:
            return None

        # landmarks only for drawing
        landmarks, hands_results, pose_results = self.extract_landmarks_from_frame(frame)

        # draw only
        frame = self.draw_landmarks(frame, hands_results, pose_results)

        # FPS display
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            fps = 30 / ((current_time - self.fps_timer) / cv2.getTickFrequency())
            self.fps_timer = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (540, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        return frame


    def get_current_predictions(self):
        """Get current predictions thread-safely"""
        with self.lock:
            return self.current_prediction, self.top3_predictions


# Global instance
recognizer = None


def get_recognizer():
    """Get or create the global recognizer instance"""
    global recognizer
    if recognizer is None:
        try:
            recognizer = RealTimeSignLanguageRecognizer()
            print("✓ Recognizer created successfully with 297 features!")
        except Exception as e:
            print(f"✗ Error creating recognizer: {e}")
            import traceback
            traceback.print_exc()
            return None
    return recognizer
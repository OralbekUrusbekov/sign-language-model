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
        print("=" * 60)
        print("Initializing RealTimeSignLanguageRecognizer")
        print("=" * 60)

        # Модель файлын тексеру
        self.model_available = os.path.exists(model_path)
        self.prediction_history = deque(maxlen=3)  # 3 кадр тарихы

        if self.model_available:
            try:
                self.model = load_model(model_path)
                print(f"✓ Model loaded: {model_path}")
                if hasattr(self.model, 'input_shape'):
                    print(f"✓ Model expects input shape: {self.model.input_shape}")
            except Exception as e:
                print(f"✗ Model load error: {e}")
                self.model = None
                self.model_available = False
        else:
            print(f"⚠ Model file not found: {model_path}")
            self.model = None

        # Scaler жүктеу
        try:
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            if self.scaler:
                print(f"✓ Scaler loaded: expects {self.scaler.n_features_in_} features")
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
                print(f"✓ Loaded feature order with {len(self.feature_order)} features")
            except Exception as e:
                print(f"✗ Feature order load error: {e}")
                self.feature_order = [f"f_{i}" for i in range(99)]
        else:
            self.feature_order = [f"f_{i}" for i in range(99)]
            print(f"Created default feature order with 99 features")

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        self.pose = None

        # === МАҢЫЗДЫ: МОДЕЛЬГЕ СӘЙКЕС ПАРАМЕТРЛЕР ===
        self.fps = 15
        self.window_sec = 1.0  # 1 секунд
        self.window_size = 3  # 3 кадр
        self.sequence_length = 15  # 15 windows = 45 кадр (3 секунд)

        # Камера индексін таңдау
        if cam_id is None:
            self.CAM_ID = self.find_working_camera()
        else:
            self.CAM_ID = cam_id

        # Buffers
        self.frame_buffer = deque(maxlen=self.window_size)
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.raw_landmarks_buffer = deque(maxlen=self.sequence_length)

        # Landmark indices
        self.pose_landmarks = list(range(33))

        # For FPS calculation
        self.frame_count = 0
        self.fps_timer = cv2.getTickCount()
        self.processing_times = deque(maxlen=30)

        # Video capture
        self.cap = None
        self.is_running = False

        # Current predictions
        self.current_prediction = None
        self.top3_predictions = None
        self.lock = threading.Lock()

        # CONFIDENCE THRESHOLD (төмендетілді)
        self.confidence_threshold = 0.15

        print(f"\n✓ Initialization complete!")
        print(f"📊 Config: window_size={self.window_size}, sequence_length={self.sequence_length}")
        print(f"📊 Total frames needed: {self.window_size * self.sequence_length} frames")
        print(f"📊 Time needed: {(self.window_size * self.sequence_length) / self.fps:.1f} seconds")
        print("=" * 60)

    def find_working_camera(self, max_check=5):
        """Жұмыс істейтін камераны табу"""
        print("\n🔍 Looking for working cameras...")

        for i in range(max_check):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"  ✓ Camera {i} is WORKING")
                        cap.release()
                        return i
                    cap.release()
            except:
                pass

        print("  ⚠ Using default camera 0")
        return 0

    def _init_mediapipe(self):
        """Initialize MediaPipe models with optimized settings"""
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.3,  # Төмендетілді
                min_tracking_confidence=0.3
            )

        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
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
                self.cap = cv2.VideoCapture(self.CAM_ID)

                if not self.cap.isOpened():
                    print("Failed to open camera, trying index 0...")
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        return False

                # Camera settings for speed
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            self._init_mediapipe()
            self.frame_buffer.clear()
            self.feature_buffer.clear()
            self.raw_landmarks_buffer.clear()
            self.prediction_history.clear()
            self.is_running = True
            print(f"✓ Camera started successfully")
            return True

        except Exception as e:
            print(f"✗ Error starting capture: {e}")
            return False

    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        self._cleanup_mediapipe()
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.current_prediction = None
            self.top3_predictions = None

    def extract_landmarks_from_frame(self, frame):
        """Extract ONLY POSE landmarks (WSASL100 uses pose only)"""
        if self.hands is None or self.pose is None:
            return {"pose": []}, None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)

        # Hands still processed for visualization only
        hands_results = self.hands.process(frame_rgb)

        landmarks = {
            "hand_0": [],
            "hand_1": [],
            "hand_labels": [],
            "pose": [],
        }

        # Extract POSE landmarks (33 points) - MAIN FEATURES
        if pose_results.pose_landmarks:
            for idx in range(33):
                lm = pose_results.pose_landmarks.landmark[idx]
                landmarks["pose"].append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 1.0
                })
        else:
            # No pose detected - add zeros
            for _ in range(33):
                landmarks["pose"].append({"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0})

        # Extract HAND landmarks (for visualization only)
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if hand_idx > 1:
                    break

                label = hands_results.multi_handedness[hand_idx].classification[0].label
                landmarks["hand_labels"].append(label)

                for lm in hand_landmarks.landmark:
                    landmarks[f"hand_{hand_idx}"].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    })

        return landmarks, hands_results, pose_results

    def extract_window_features(self, frames):
        """
        Extract 99 features from 5 frames (pose only)
        Returns: 99 features (33 landmarks * 3 coordinates)
        """
        features = []

        for landmark_idx in range(33):  # 33 pose landmarks
            for coord_idx, coord in enumerate(['x', 'y', 'z']):
                values = []

                # Collect values from all 5 frames
                for frame_idx, frame in enumerate(frames):
                    if landmark_idx < len(frame["pose"]):
                        values.append(frame["pose"][landmark_idx][coord])
                    else:
                        values.append(0.0)

                # Calculate MEAN value (what model expects)
                if values:
                    # Weight recent frames more
                    weights = [0.5, 0.8, 1.0, 1.2, 1.5]  # last frame gets highest weight
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    total_weight = sum(weights)
                    mean_val = weighted_sum / total_weight
                else:
                    mean_val = 0.0

                features.append(mean_val)

        return np.array(features)  # Shape: (99,)

    def predict(self):
        """Make prediction with 99 features"""
        if not self.model_available or self.model is None:
            return None, None

        if len(self.feature_buffer) < self.sequence_length:
            return None, None

        try:
            # Prepare data (sequence_length, 99)
            X = np.array(list(self.feature_buffer))
            print(f"📊 Feature buffer shape: {X.shape}")  # (15, 99)

            # Reshape for model (1, sequence_length, 99)
            X = X.reshape(1, self.sequence_length, 99)
            print(f"📊 Reshaped for model: {X.shape}")  # (1, 15, 99)

            # Scale if scaler available
            if self.scaler:
                X_reshaped = X.reshape(-1, 99)
                X_scaled = self.scaler.transform(X_reshaped)
                X = X_scaled.reshape(1, self.sequence_length, 99)

            # Predict
            predictions = self.model.predict(X, verbose=0)[0]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions)[-3:][::-1]

            if self.label_encoder:
                top3_labels = self.label_encoder.inverse_transform(top3_indices)
            else:
                top3_labels = [f"class_{i}" for i in top3_indices]

            top3_confidences = predictions[top3_indices]

            # Apply confidence threshold
            if top3_confidences[0] < self.confidence_threshold:
                predicted_label = None
            else:
                predicted_label = top3_labels[0]

                # Smooth predictions with history
                self.prediction_history.append(predicted_label)
                if len(self.prediction_history) == 3:
                    from collections import Counter
                    predicted_label = Counter(self.prediction_history).most_common(1)[0][0]

            return predicted_label, list(zip(top3_labels, top3_confidences * 100))

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def draw_landmarks(self, frame, hands_results, pose_results):
        """Draw landmarks on frame"""
        # Draw pose landmarks (always)
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Draw hand landmarks
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        return frame

    def process_frame(self, frame):
        """Process a single frame for prediction"""
        if not self.is_running:
            return frame

        start_time = time.time()

        # Extract landmarks
        landmarks, hands_results, pose_results = self.extract_landmarks_from_frame(frame)

        # Store raw landmarks for visualization
        self.raw_landmarks_buffer.append(landmarks)

        # Add to frame buffer
        self.frame_buffer.append(landmarks)

        # If we have enough frames, extract window features
        if len(self.frame_buffer) == self.window_size:
            window_features = self.extract_window_features(list(self.frame_buffer))
            self.feature_buffer.append(window_features)
            self.frame_buffer.clear()  # Clear for next window

        # Make prediction if we have enough windows
        if len(self.feature_buffer) == self.sequence_length:
            pred, top3 = self.predict()
            with self.lock:
                self.current_prediction = pred
                self.top3_predictions = top3

        # Draw landmarks
        frame = self.draw_landmarks(frame, hands_results, pose_results)

        # Calculate and display FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            current_time = cv2.getTickCount()
            fps = 10 / ((current_time - self.fps_timer) / cv2.getTickFrequency())
            self.fps_timer = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Show buffer status
        buffer_status = f"Frames: {len(self.frame_buffer)}/{self.window_size} | Windows: {len(self.feature_buffer)}/{self.sequence_length}"
        cv2.putText(
            frame,
            buffer_status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        # Show current prediction
        with self.lock:
            if self.current_prediction:
                cv2.putText(
                    frame,
                    f"Pred: {self.current_prediction}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        return frame

    def get_frame(self):
        """Get processed frame for streaming"""
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

        # Process frame
        frame = self.process_frame(frame)

        return frame

    def get_current_predictions(self):
        """Get current predictions thread-safely"""
        with self.lock:
            return self.current_prediction, self.top3_predictions

    def get_landmarks_for_display(self):
        """Get latest landmarks for React Native display"""
        if len(self.raw_landmarks_buffer) > 0:
            return self.raw_landmarks_buffer[-1]
        return {"hand_0": [], "hand_1": [], "hand_labels": [], "pose": []}


# Global instance
recognizer = None


def get_recognizer():
    """Get or create the global recognizer instance"""
    global recognizer
    if recognizer is None:
        try:
            recognizer = RealTimeSignLanguageRecognizer()
        except Exception as e:
            print(f"Error creating recognizer: {e}")
            return None
    return recognizer
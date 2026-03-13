#!/usr/bin/env python3
"""
Real-time Sign Language Recognition using MediaPipe and LSTM model
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import joblib
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import os

# Keras үйлесімділік мәселесін шешу
os.environ['TF_USE_LEGACY_KERAS'] = '1'

warnings.filterwarnings("ignore")


class RealTimeSignLanguageRecognizer:
    def __init__(
            self,
            model_path="app/model/sign_language_recognition_fixed.keras",  # Өзгертілді
            scaler_path="app/model/scaler.pkl",
            label_encoder_path="app/model/label_encoder.pkl",
            feature_order_path="app/model/feature_order.json",
            cam_id=0,
    ):
        # Load model and preprocessing tools
        print("Loading model and preprocessing tools...")

        # Модельді жүктеу
        try:
            self.model = load_model(model_path)
            print(f"Модель жүктелді: {model_path}")
        except Exception as e:
            print(f"Модельді жүктеу қатесі: {e}")
            # Егер .h5 форматында болса
            try:
                h5_path = model_path.replace('.keras', '.h5')
                self.model = load_model(h5_path)
                print(f"Модель .h5 форматынан жүктелді: {h5_path}")
            except Exception as e2:
                print(f".h5 форматында да жүктеу сәтсіз: {e2}")
                raise

        # Модель архитектурасын көрсету
        self.model.summary()

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)

        with open(feature_order_path, "r") as f:
            self.feature_order = json.load(f)

        # Қалған код өзгеріссіз...
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Parameters
        self.fps = 15
        self.window_sec = 0.5
        self.window_size = int(self.fps * self.window_sec)
        self.sequence_length = 1
        self.CAM_ID = cam_id

        # Buffers
        self.frame_buffer = deque(maxlen=self.window_size)
        self.feature_buffer = deque(maxlen=self.sequence_length)

        # Landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]
        self.pose_joints = [11, 12, 13, 14, 15, 16]

        # For FPS calculation
        self.frame_count = 0
        self.fps_timer = cv2.getTickCount()

        print("Initialization complete!")

    # Қалған әдістер өзгеріссіз қалады
    def extract_landmarks_from_frame(self, frame):
        # ... (өзгеріссіз)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        hands_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        landmarks = {
            "hand_0": [],  # First detected hand (could be left or right)
            "hand_1": [],  # Second detected hand (could be left or right)
            "pose": [],
        }

        # Extract hand landmarks - use detection order as in training
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(
                    hands_results.multi_hand_landmarks
            ):
                if hand_idx > 1:  # Only process first 2 hands
                    break

                # Extract fingertip landmarks
                for lm_idx in self.finger_tips:
                    lm = hand_landmarks.landmark[lm_idx]
                    landmarks[f"hand_{hand_idx}"].append(
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                    )

        # Extract pose landmarks
        if pose_results.pose_landmarks:
            for idx in self.pose_joints:
                lm = pose_results.pose_landmarks.landmark[idx]
                landmarks["pose"].append({"x": lm.x, "y": lm.y, "z": lm.z})

        return landmarks, hands_results, pose_results

    def extract_window_features(self, window_landmarks):
        # ... (өзгеріссіз)
        features = {}

        # Initialize all features to 0
        for feature_name in self.feature_order:
            features[feature_name] = 0.0

        # Pose features (joints)
        pose_coords = {"x": [], "y": [], "z": []}
        for frame_landmarks in window_landmarks:
            for lm in frame_landmarks["pose"]:
                pose_coords["x"].append(lm["x"])
                pose_coords["y"].append(lm["y"])
                pose_coords["z"].append(lm["z"])

        for axis in ["x", "y", "z"]:
            if pose_coords[axis]:
                features[f"joints_{axis}_mean"] = float(np.mean(pose_coords[axis]))
                features[f"joints_{axis}_std"] = float(np.std(pose_coords[axis]))

        # Hand features - matching training data convention
        # hand_0 -> left features, hand_1 -> right features
        for hand_idx, feature_prefix in [(0, "left"), (1, "right")]:
            hand_coords = {"x": [], "y": [], "z": []}
            for frame_landmarks in window_landmarks:
                for lm in frame_landmarks[f"hand_{hand_idx}"]:
                    hand_coords["x"].append(lm["x"])
                    hand_coords["y"].append(lm["y"])
                    hand_coords["z"].append(lm["z"])

            for axis in ["x", "y", "z"]:
                if hand_coords[axis]:
                    features[f"{feature_prefix}_tips_{axis}_mean"] = float(
                        np.mean(hand_coords[axis])
                    )
                    features[f"{feature_prefix}_tips_{axis}_std"] = float(
                        np.std(hand_coords[axis])
                    )

        # Ensure features are in correct order
        ordered_features = [features[name] for name in self.feature_order]
        return np.array(ordered_features)

    def predict(self):
        # ... (өзгеріссіз)
        if len(self.feature_buffer) < self.sequence_length:
            return None, None

        # Prepare input
        X = np.array(list(self.feature_buffer))
        X = X.reshape(1, self.sequence_length, -1)

        # Scale features
        X_reshaped = X.reshape(X.shape[1], -1)
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(1, X.shape[1], -1)

        # Predict
        predictions = self.model.predict(X, verbose=0)[0]

        # Get top 3 predictions
        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3_labels = self.label_encoder.inverse_transform(top3_indices)
        top3_confidences = predictions[top3_indices]

        predicted_class = top3_indices[0]
        predicted_label = top3_labels[0]

        return predicted_label, list(zip(top3_labels, top3_confidences * 100))

    def draw_landmarks(self, frame, hands_results, pose_results):
        # ... (өзгеріссіз)
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            # Only draw the joints we use
            for idx in self.pose_joints:
                lm = pose_results.pose_landmarks.landmark[idx]
                h, w = frame.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    def run(self):
        # ... (өзгеріссіз)
        cap = cv2.VideoCapture(self.CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\nStarting real-time sign language recognition...")
        print("Press 'q' to quit\n")

        frame_skip = max(1, int(cap.get(cv2.CAP_PROP_FPS) / self.fps))
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # Skip frames to match target FPS
            if frame_counter % frame_skip != 0:
                continue

            # Don't mirror the frame - process as-is to match training data
            # frame = cv2.flip(frame, 1)

            # Extract landmarks
            landmarks, hands_results, pose_results = self.extract_landmarks_from_frame(
                frame
            )

            # Add to buffer
            self.frame_buffer.append(landmarks)

            # Extract features when window is full
            if len(self.frame_buffer) == self.window_size:
                features = self.extract_window_features(list(self.frame_buffer))
                self.feature_buffer.append(features)
                self.frame_buffer.clear()

            # Make prediction
            predicted_label, top3 = self.predict()

            # Draw landmarks
            frame = self.draw_landmarks(frame, hands_results, pose_results)

            # Display prediction on frame
            if predicted_label:
                # Main prediction
                cv2.putText(
                    frame,
                    f"Prediction: {predicted_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Top 3 predictions
                y_offset = 60
                for i, (label, conf) in enumerate(top3):
                    text = f"{i + 1}. {label}: {conf:.1f}%"
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                # Print to terminal
                print(
                    f"\rPredicted: {predicted_label} | Top 3: "
                    + " | ".join([f"{l}: {c:.1f}%" for l, c in top3]),
                    end="",
                    flush=True,
                )
            else:
                cv2.putText(
                    frame,
                    "Collecting frames...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Calculate and display FPS
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

            # Show frame
            cv2.imshow("Sign Language Recognition", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print("\n\nClosing...")
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()


def main():
    recognizer = RealTimeSignLanguageRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
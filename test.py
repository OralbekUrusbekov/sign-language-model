# test_fixed.py
import cv2
import numpy as np
import mediapipe as mp
import json
import joblib
from collections import deque
from tensorflow.keras.models import load_model


class TestRecognizer:
    def __init__(self):
        # Load model and preprocessing tools
        self.model = load_model("app/model/sign_language_recognition_fixed.keras")
        self.scaler = joblib.load("app/model/scaler.pkl")
        self.label_encoder = joblib.load("app/model/label_encoder.pkl")

        with open("app/model/feature_order.json", "r") as f:
            self.feature_order = json.load(f)

        print(f"Model classes: {len(self.label_encoder.classes_)}")
        print(f"First 10 classes: {self.label_encoder.classes_[:10]}")
        print(f"Feature dimension: {len(self.feature_order)}")

        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices - ALL landmarks, not just fingertips
        self.hand_landmarks = list(range(21))  # All 21 hand landmarks
        self.pose_landmarks = list(range(33))  # All 33 pose landmarks

        print(f"Hand landmarks: {len(self.hand_landmarks)}")
        print(f"Pose landmarks: {len(self.pose_landmarks)}")

    def extract_all_landmarks(self, frame):
        """Extract ALL landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        landmarks = {
            "hand_0": [],  # All landmarks for hand 0
            "hand_1": [],  # All landmarks for hand 1
            "pose": [],  # All pose landmarks
            "hand_labels": []
        }

        # Get hand labels (left/right)
        if hands_results.multi_handedness:
            for hand_idx, handedness in enumerate(hands_results.multi_handedness):
                if hand_idx > 1:
                    break
                label = handedness.classification[0].label.lower()
                landmarks["hand_labels"].append((hand_idx, label))

        # Extract ALL hand landmarks (0-20)
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(hands_results.multi_hand_landmarks):
                if hand_idx > 1:
                    break

                for lm_idx in range(21):  # All 21 hand landmarks
                    lm = hand_lms.landmark[lm_idx]
                    landmarks[f"hand_{hand_idx}"].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": 1.0
                    })

        # Extract ALL pose landmarks (0-32)
        if pose_results.pose_landmarks:
            for lm_idx in range(33):  # All 33 pose landmarks
                lm = pose_results.pose_landmarks.landmark[lm_idx]
                landmarks["pose"].append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        return landmarks, hands_results, pose_results

    def extract_features_single_frame(self, landmarks):
        """Extract features for a single frame (for testing)"""
        features = np.zeros(len(self.feature_order))

        # Print some debug info
        print(f"\nLandmarks found:")
        print(f"  Hand 0: {len(landmarks['hand_0'])} landmarks")
        print(f"  Hand 1: {len(landmarks['hand_1'])} landmarks")
        print(f"  Pose: {len(landmarks['pose'])} landmarks")

        # Simple approach: just flatten all coordinates
        all_coords = []

        # Hand 0 landmarks
        for lm in landmarks["hand_0"]:
            all_coords.extend([lm["x"], lm["y"], lm["z"]])

        # Hand 1 landmarks
        for lm in landmarks["hand_1"]:
            all_coords.extend([lm["x"], lm["y"], lm["z"]])

        # Pose landmarks
        for lm in landmarks["pose"]:
            all_coords.extend([lm["x"], lm["y"], lm["z"]])

        # Pad or truncate to match feature dimension
        if len(all_coords) > 0:
            if len(all_coords) > len(features):
                all_coords = all_coords[:len(features)]
            elif len(all_coords) < len(features):
                all_coords.extend([0.0] * (len(features) - len(all_coords)))

            features = np.array(all_coords)

        return features

    def test_live(self):
        """Test with live camera"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\nStarting live test...")
        print("Press 'q' to quit, 's' to capture and predict\n")

        frame_count = 0
        prediction_history = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Extract landmarks
            landmarks, hands_results, pose_results = self.extract_all_landmarks(frame)

            # Every 30 frames, make a prediction
            if frame_count % 30 == 0:
                features = self.extract_features_single_frame(landmarks)

                # Check if we have any non-zero features
                non_zero = np.count_nonzero(features)
                print(f"\nFeatures - Non-zero: {non_zero}/{len(features)}")

                if non_zero > 0:
                    # Scale and predict
                    features_scaled = self.scaler.transform([features])
                    features_scaled = features_scaled.reshape(1, 1, -1)

                    predictions = self.model.predict(features_scaled, verbose=0)[0]

                    # Get top 5 predictions
                    top5_idx = np.argsort(predictions)[-5:][::-1]

                    print("\nTop 5 predictions:")
                    for idx in top5_idx:
                        label = self.label_encoder.inverse_transform([idx])[0]
                        conf = predictions[idx] * 100
                        print(f"  {label}: {conf:.2f}%")
                        prediction_history.append(label)

            # Draw landmarks on frame
            if hands_results and hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_lms,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

            if pose_results and pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

            # Show info
            cv2.putText(
                frame,
                f"Hands: {len(landmarks['hand_0']) > 0} {len(landmarks['hand_1']) > 0}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pose: {len(landmarks['pose']) > 0}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if prediction_history:
                cv2.putText(
                    frame,
                    f"Last: {prediction_history[-1]}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            cv2.imshow("Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manual capture and predict
                features = self.extract_features_single_frame(landmarks)
                non_zero = np.count_nonzero(features)
                print(f"\nManual capture - Non-zero features: {non_zero}/{len(features)}")

                if non_zero > 10:  # At least some features detected
                    features_scaled = self.scaler.transform([features])
                    features_scaled = features_scaled.reshape(1, 1, -1)
                    predictions = self.model.predict(features_scaled, verbose=0)[0]

                    print("\nPredictions:")
                    for idx in np.argsort(predictions)[-5:][::-1]:
                        label = self.label_encoder.inverse_transform([idx])[0]
                        conf = predictions[idx] * 100
                        print(f"  {label}: {conf:.2f}%")
                else:
                    print("No landmarks detected! Make sure your hand/body is visible.")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()


if __name__ == "__main__":
    tester = TestRecognizer()
    tester.test_live()
import os
from datetime import datetime

from flask import (
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
    current_app as app,
    Response,
)
from celery.result import AsyncResult
import cv2

from app import db
from app.models import Video, FrameLandmark
from app.tasks import process_video_landmarks, celery
from app.config import VIDEO_DIR
from app.utils.real_time_recognition import get_recognizer
import numpy as np


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """FIXED: React Native frame processing with 225 features"""
    print("\n" + "=" * 60)
    print("🔵 PREDICT_FRAME called - WLASL100 VERSION")
    print("=" * 60)

    try:
        # Get file from request
        if 'image' in request.files:
            file = request.files['image']
            print("✅ Using 'image' field")
        elif 'frame' in request.files:
            file = request.files['frame']
            print("✅ Using 'frame' field")
        else:
            print(f"❌ No file found. Keys: {list(request.files.keys())}")
            return jsonify({
                'status': 'error',
                'message': 'No image or frame provided'
            }), 400

        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'}), 400

        # Get recognizer
        recognizer = get_recognizer()
        if recognizer is None:
            return jsonify({'status': 'error', 'message': 'Recognizer not initialized'}), 500

        # Extract landmarks (pose + hands)
        landmarks, hands_results, pose_results = recognizer.extract_landmarks_from_frame(frame)

        # Check if pose detected
        has_pose = len(landmarks["pose"]) > 0 and any(
            lm["x"] != 0 or lm["y"] != 0 or lm["z"] != 0 for lm in landmarks["pose"]
        )

        # Check if hands detected
        has_hands = len(landmarks["hand_0"]) > 0 or len(landmarks["hand_1"]) > 0

        if not has_pose:
            print("⚠ No pose detected")
            return jsonify({
                'status': 'waiting',
                'message': 'No pose detected',
                'current_prediction': None,
                'top3': [],
                'landmarks_detected': {
                    'pose': False,
                    'hands': has_hands
                }
            })

        # Extract features (225 features)
        features = recognizer.extract_raw_features(landmarks)
        recognizer.feature_buffer.append(features)

        # Buffer status
        buffer_len = len(recognizer.feature_buffer)

        # Prepare landmarks for React Native
        serializable_landmarks = {
            'hand_0': [{'x': p.get('x', 0), 'y': p.get('y', 0), 'z': p.get('z', 0)}
                       for p in landmarks.get('hand_0', [])[:21]],
            'hand_1': [{'x': p.get('x', 0), 'y': p.get('y', 0), 'z': p.get('z', 0)}
                       for p in landmarks.get('hand_1', [])[:21]],
            'hand_labels': landmarks.get('hand_labels', []),
            'pose': [{'x': p.get('x', 0), 'y': p.get('y', 0), 'z': p.get('z', 0)}
                     for p in landmarks.get('pose', [])[:33]]
        }

        # Predict when buffer is full
        if buffer_len >= recognizer.sequence_length:
            print(f"✅ Buffer full! Predicting with {buffer_len} frames...")
            print(f"📊 Features shape: {features.shape}")
            print(f"📊 Buffer size: {len(recognizer.feature_buffer)}")

            predicted_label, top3 = recognizer.predict()

            if predicted_label:
                print(f"🎯 PREDICTION: {predicted_label}")
                return jsonify({
                    'status': 'success',
                    'current_prediction': predicted_label,
                    'top3': [{'label': l, 'confidence': float(c)} for l, c in top3] if top3 else [],
                    'landmarks': serializable_landmarks,
                    'buffer_status': {
                        'frames': buffer_len,
                        'needed': recognizer.sequence_length
                    },
                    'landmarks_detected': {
                        'pose': has_pose,
                        'hands': has_hands
                    }
                })
            else:
                return jsonify({
                    'status': 'waiting',
                    'current_prediction': None,
                    'top3': [{'label': l, 'confidence': float(c)} for l, c in top3] if top3 else [],
                    'landmarks': serializable_landmarks,
                    'message': 'Low confidence prediction',
                    'buffer_status': {
                        'frames': buffer_len,
                        'needed': recognizer.sequence_length
                    },
                    'landmarks_detected': {
                        'pose': has_pose,
                        'hands': has_hands
                    }
                })

        # Still collecting frames
        return jsonify({
            'status': 'waiting',
            'message': f'Collecting frames: {buffer_len}/{recognizer.sequence_length}',
            'buffer_status': {
                'frames': buffer_len,
                'needed': recognizer.sequence_length
            },
            'current_prediction': None,
            'top3': [],
            'landmarks': serializable_landmarks,
            'landmarks_detected': {
                'pose': has_pose,
                'hands': has_hands
            }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'current_prediction': None,
            'top3': []
        }), 500


@app.route("/")
def index():
    return render_template("index.html")


# --- Video Streaming for Real-Time Recognition ---
def generate_frames():
    """Generate frames for video streaming"""
    recognizer = get_recognizer()

    while True:
        frame = recognizer.get_frame()
        if frame is None:
            frame = recognizer.get_frame()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/current_predictions')
def current_predictions():
    """Get current predictions as JSON"""
    recognizer = get_recognizer()
    current, top3 = recognizer.get_current_predictions()

    if current is None:
        return jsonify({
            'status': 'waiting',
            'message': 'Collecting frames...'
        })

    return jsonify({
        'status': 'success',
        'current_prediction': current,
        'top3': [
            {'label': label, 'confidence': float(conf)}
            for label, conf in top3
        ]
    })


@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    """Start or stop recognition"""
    action = request.json.get('action', 'start')
    recognizer = get_recognizer()

    if action == 'start':
        success = recognizer.start_capture()
        if success:
            return jsonify({'status': 'started', 'success': True})
        else:
            return jsonify({'status': 'error', 'success': False, 'message': 'Failed to start camera'})
    else:
        recognizer.stop_capture()
        return jsonify({'status': 'stopped', 'success': True})


@app.route('/recognition_status')
def recognition_status():
    """Get current recognition status"""
    recognizer = get_recognizer()
    return jsonify({
        'is_running': recognizer.is_running
    })


@app.route('/reset_buffer', methods=['POST'])
def reset_buffer():
    """Reset buffer for new gesture"""
    recognizer = get_recognizer()
    recognizer.reset_buffers()
    return jsonify({
        'status': 'success',
        'message': 'Buffer reset successfully'
    })


# --- Video management routes (unchanged) ---
def _save_video_file(label, video_file):
    """Saves uploaded video to disk and creates a Video record"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(VIDEO_DIR, label)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{timestamp}.webm"
    video_path = os.path.join(save_dir, filename)
    video_file.save(video_path)

    video = Video(path=video_path, label=label)
    db.session.add(video)
    db.session.commit()
    return video


@app.route("/record", methods=["GET"])
def record_page():
    videos = Video.query.order_by(Video.id.desc()).all()
    video_list = []
    for v in videos:
        rel_path = os.path.relpath(v.path, os.path.join(app.root_path, "videos"))
        video_list.append(
            {
                "id": v.id,
                "label": v.label,
                "filename": os.path.basename(v.path),
                "url": url_for("serve_video", filename=rel_path),
            }
        )
    return render_template("save_video.html", videos=video_list)


@app.route("/import", methods=["GET"])
def import_page():
    return render_template("import.html")


@app.route("/import", methods=["POST"])
def import_video():
    label = request.form.get("label", "").strip()
    if not label:
        return jsonify({"status": "error", "message": "Label is required."}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"status": "error", "message": "No video file uploaded."}), 400

    video = _save_video_file(label, video_file)

    rel_path = os.path.relpath(video.path, os.path.join(app.root_path, "videos"))
    video_url = url_for("serve_video", filename=rel_path)

    mirror = request.form.get("mirror", "true").lower() == "true"
    task = process_video_landmarks.apply_async(
        args=[video.id], kwargs={"mirror": mirror}
    )

    return jsonify(
        {
            "status": "success",
            "message": "Import started.",
            "video_id": video.id,
            "video_url": video_url,
            "task_id": task.id,
        }
    )


@app.route("/save_video", methods=["POST"])
def save_video():
    label = request.form.get("label", "").strip()
    if not label:
        return jsonify({"status": "error", "message": "Label can not be empty."}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"status": "error", "message": "No video uploaded."}), 400

    video = _save_video_file(label, video_file)

    rel_path = os.path.relpath(video.path, os.path.join(app.root_path, "videos"))
    video_url = url_for("serve_video", filename=rel_path)

    return jsonify(
        {
            "status": "success",
            "message": "Video saved successfully.",
            "video_id": video.id,
            "path": video_url,
        }
    )


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(os.path.join(app.root_path, "videos"), filename)


@app.route("/gallery", methods=["GET"])
def gallery():
    videos = Video.query.order_by(Video.id.desc()).all()
    video_list = []
    for v in videos:
        rel_path = os.path.relpath(v.path, os.path.join(app.root_path, "videos"))
        video_list.append(
            {
                "id": v.id,
                "label": v.label,
                "filename": os.path.basename(v.path),
                "url": url_for("serve_video", filename=rel_path),
            }
        )
    return render_template("gallery.html", videos=video_list)


@app.route("/process_video", methods=["POST"])
def process_video():
    video_id = request.form.get("video_id", type=int)
    mirror = request.form.get("mirror", "true").lower() == "true"
    start_time = request.form.get("start_time", type=float, default=0.0)
    end_time = request.form.get("end_time", type=float, default=0.0)

    if start_time >= end_time:
        return jsonify(
            {"status": "error", "message": "Start time must be less than end time."}
        ), 400
    if start_time < 0 or end_time < 0:
        return jsonify(
            {"status": "error", "message": "Start and end time must be positive."}
        ), 400
    if start_time == end_time:
        return jsonify(
            {"status": "error", "message": "Start time and end time must be different."}
        ), 400

    video = Video.query.get(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video not found."}), 404

    task = process_video_landmarks.apply_async(
        args=[video_id],
        kwargs={
            "mirror": mirror,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    return jsonify(
        {
            "status": "success",
            "task_id": task.id,
            "message": "Landmark extraction started.",
        }
    )


@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        response = {
            "state": task.state,
            "current": 0,
            "total": 1,
            "status": "Pending...",
        }
    elif task.state == "PROGRESS":
        response = {
            "state": task.state,
            "current": task.info.get("current", 0),
            "total": task.info.get("total", 1),
            "status": task.info.get("status", ""),
        }
    elif task.state == "SUCCESS":
        response = {
            "state": task.state,
            "current": task.info.get("current", 0),
            "total": task.info.get("total", 1),
            "status": "Completed",
        }
    else:
        response = {
            "state": task.state,
            "current": 0,
            "total": 1,
            "status": str(task.info),
        }

    return jsonify(response)


@app.route("/delete_video/<int:video_id>", methods=["DELETE"])
def delete_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video not found."}), 404

    try:
        if os.path.exists(video.path):
            os.remove(video.path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"File delete error: {e}"}), 500

    try:
        FrameLandmark.query.filter_by(video_id=video_id).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify(
            {"status": "error", "message": f"Database delete error: {e}"}
        ), 500

    try:
        db.session.delete(video)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify(
            {"status": "error", "message": f"Database delete error: {e}"}
        ), 500

    return jsonify(
        {"status": "success", "message": "Video and related records deleted."}
    )


@app.route("/edit_video", methods=["POST"])
def edit_video():
    data = request.get_json() or {}
    video_id = data.get("video_id")
    new_label = data.get("label", "").strip()
    start_time = data.get("start_time", 0.0)
    end_time = data.get("end_time", 0.0)
    mirror = data.get("mirror", True)

    if not video_id or new_label == "":
        return jsonify(
            {"status": "error", "message": "video_id and non-empty label required."}
        ), 400
    if start_time >= end_time or start_time < 0 or end_time < 0:
        return jsonify({"status": "error", "message": "Invalid time range."}), 400

    video = Video.query.get(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video not found."}), 404

    video.label = new_label
    db.session.commit()

    FrameLandmark.query.filter_by(video_id=video_id).delete()
    db.session.commit()

    task = process_video_landmarks.apply_async(
        args=[video_id],
        kwargs={"mirror": mirror, "start_time": start_time, "end_time": end_time},
    )

    return jsonify(
        {"status": "success", "message": "Re-extraction started.", "task_id": task.id}
    )

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
from app.models import Video, FrameLandmark  # DB Models
from app.tasks import process_video_landmarks, celery
from app.config import VIDEO_DIR
from app.utils.real_time_recognition import get_recognizer
import numpy as np


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """React Native-тен келген бір кадрды өңдеу - 99 FEATURES"""
    print("\n" + "=" * 50)
    print("🔵 PREDICT_FRAME called (99 features)")

    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        recognizer = get_recognizer()

        # БАРЛЫҚ landmark-тарды экстракциялау
        landmarks, hands_results, pose_results = recognizer.extract_landmarks_from_frame(frame)

        # Буферге қосу
        recognizer.frame_buffer.append(landmarks)

        frame_buffer_len = len(recognizer.frame_buffer)
        feature_buffer_len = len(recognizer.feature_buffer)

        print(f"📊 Frame buffer: {frame_buffer_len}/{recognizer.window_size}")
        print(f"📊 Feature buffer: {feature_buffer_len}/{recognizer.sequence_length}")

        response_data = {
            'status': 'waiting',
            'message': f'Collecting frames... ({frame_buffer_len}/{recognizer.window_size})',
            'frame_buffer': frame_buffer_len,
            'feature_buffer': feature_buffer_len,
            'landmarks': landmarks
        }

        # Егер терезе толса, 99 feature экстракциясы
        from collections import deque

        if len(recognizer.frame_buffer) >= recognizer.window_size:
            print("✅ Window full! Extracting 99 features...")

            frames_to_process = list(recognizer.frame_buffer)
            features = recognizer.extract_window_features(frames_to_process)

            recognizer.feature_buffer.append(features)

            recognizer.frame_buffer = deque(
                list(recognizer.frame_buffer)[-(recognizer.window_size - 1):],
                maxlen=recognizer.window_size
            )

            if len(recognizer.feature_buffer) > recognizer.sequence_length:
                recognizer.feature_buffer = recognizer.feature_buffer[-recognizer.sequence_length:]

            feature_buffer_len = len(recognizer.feature_buffer)

            print(f"📊 Feature buffer after: {feature_buffer_len}/{recognizer.sequence_length}")

            response_data['message'] = f'Features extracted ({feature_buffer_len}/{recognizer.sequence_length})'
        # Егер жеткілікті терезе болса (5), болжау
        if len(recognizer.feature_buffer) >= recognizer.sequence_length:
            print(f"✅ Predicting with 5 windows of 99 features...")
            predicted_label, top3 = recognizer.predict()

            if predicted_label:
                response_data = {
                    'status': 'success',
                    'current_prediction': predicted_label,
                    'top3': [{'label': l, 'confidence': float(c)} for l, c in top3],
                    'feature_count': 99,
                    'windows': len(recognizer.feature_buffer),
                    'landmarks': landmarks
                }
                print(f"🎯 Prediction: {predicted_label}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
            # Send a black frame if no frame available
            frame = recognizer.get_frame()  # Will return placeholder

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/cameras', methods=['GET'])
def list_cameras():
    """Қолжетімді камералар тізімін қайтару"""
    recognizer = get_recognizer()
    cameras = recognizer.get_available_cameras()
    return jsonify({
        'status': 'success',
        'cameras': cameras,
        'current_camera': recognizer.CAM_ID
    })


@app.route('/camera/set', methods=['POST'])
def set_camera():
    """Камера индексін өзгерту"""
    data = request.get_json() or {}
    cam_id = data.get('cam_id')

    if cam_id is None:
        return jsonify({'status': 'error', 'message': 'cam_id required'}), 400

    recognizer = get_recognizer()

    # Егер тану жұмыс істеп тұрса, тоқтату
    was_running = recognizer.is_running
    if was_running:
        recognizer.stop_capture()

    # Камераны өзгерту
    success = recognizer.set_camera_index(cam_id)

    if not success:
        return jsonify({
            'status': 'error',
            'message': f'Camera {cam_id} is not available'
        }), 400

    # Егер бұрын жұмыс істеп тұрса, қайта бастау
    if was_running:
        recognizer.start_capture()

    return jsonify({
        'status': 'success',
        'cam_id': cam_id,
        'message': f'Camera set to index {cam_id}'
    })


@app.route('/camera/test/<int:cam_id>', methods=['GET'])
def test_camera(cam_id):
    """Бір камераны тексеру"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return jsonify({
            'status': 'error',
            'message': f'Camera {cam_id} cannot be opened'
        })

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        return jsonify({
            'status': 'success',
            'cam_id': cam_id,
            'frame_size': frame.shape
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Camera {cam_id} opened but cannot read frame'
        })


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


# --- Helper for saving videos to DB and filesystem ---
def _save_video_file(label, video_file):
    """
    Saves uploaded video to disk and creates a Video record in the database.
    Returns the Video instance.
    """
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
    """
    Renders the record page, pulling video entries directly from the database
    to include id, label, filename, and URL without extra filesystem queries.
    """
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
    """
    Handles external video upload and kicks off landmark+feature extraction.
    """
    label = request.form.get("label", "").strip()
    if not label:
        return jsonify({"status": "error", "message": "Label is required."}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"status": "error", "message": "No video file uploaded."}), 400

    # reuse helper to save file & DB record
    video = _save_video_file(label, video_file)

    # video path
    rel_path = os.path.relpath(video.path, os.path.join(app.root_path, "videos"))
    video_url = url_for("serve_video", filename=rel_path)

    # mirror setting
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


# for submission of video files
@app.route("/videos/<path:filename>")
def serve_video(filename):
    # 'videos' klasörünü uygulamanın root klasörü altında varsayılıyor
    return send_from_directory(os.path.join(app.root_path, "videos"), filename)


@app.route("/gallery", methods=["GET"])
def gallery():
    """
    Renders the gallery page using the same DB-driven list as record_page for consistency.
    """
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
    """
    AJAX ile çağrılan endpoint: video_id alıp arka planda Celery task'ını başlatır.
    """
    video_id = request.form.get("video_id", type=int)
    # get mirror value from form (default: true)
    mirror = request.form.get("mirror", "true").lower() == "true"
    # get start_time and end_time from form
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

    # start the Celery task and send video_id and mirror value
    # to the task
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
        # Görev henüz başlatılmamış veya sonuç kaydı yok
        response = {
            "state": task.state,
            "current": 0,
            "total": 1,
            "status": "Pending...",
        }
    elif task.state == "PROGRESS":
        # PROGRESS durumunda, görev ilerleme verisi (current, total) içerir
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
        # FAILURE veya başka bir durum
        response = {
            "state": task.state,
            "current": 0,
            "total": 1,
            "status": str(task.info),
        }

    return jsonify(response)


# DELETE endpoint: remove video file and related DB records
@app.route("/delete_video/<int:video_id>", methods=["DELETE"])
def delete_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video not found."}), 404

    # 1. Delete file from disk
    try:
        if os.path.exists(video.path):
            os.remove(video.path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"File delete error: {e}"}), 500

    # 2. Delete related landmarks
    try:
        FrameLandmark.query.filter_by(video_id=video_id).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify(
            {"status": "error", "message": f"Database delete error: {e}"}
        ), 500

    # 3. Delete Video record
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


# EDIT endpoint: update label and re-extract landmarks/features
@app.route("/edit_video", methods=["POST"])
def edit_video():
    data = request.get_json() or {}
    video_id = data.get("video_id")
    new_label = data.get("label", "").strip()
    start_time = data.get("start_time", 0.0)
    end_time = data.get("end_time", 0.0)
    mirror = data.get("mirror", True)

    # basic validation
    if not video_id or new_label == "":
        return jsonify(
            {"status": "error", "message": "video_id and non-empty label required."}
        ), 400
    if start_time >= end_time or start_time < 0 or end_time < 0:
        return jsonify({"status": "error", "message": "Invalid time range."}), 400

    video = Video.query.get(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video not found."}), 404

    # 1. Update label
    video.label = new_label
    db.session.commit()

    # 2. Remove old FrameLandmark entries
    FrameLandmark.query.filter_by(video_id=video_id).delete()
    db.session.commit()

    # 3. Trigger Celery task for fresh extraction
    task = process_video_landmarks.apply_async(
        args=[video_id],
        kwargs={"mirror": mirror, "start_time": start_time, "end_time": end_time},
    )

    return jsonify(
        {"status": "success", "message": "Re-extraction started.", "task_id": task.id}
    )
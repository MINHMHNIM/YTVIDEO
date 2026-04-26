import os
import cv2
import subprocess
from collections import deque
from dotenv import load_dotenv

load_dotenv()
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')
FFMPEG_BIN = os.path.join(FFMPEG_PATH, 'ffmpeg.exe') if FFMPEG_PATH else 'ffmpeg'

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def is_anchor_face(x, y, fw, fh, w, h):
    """
    Phán đoán có phải mặt phát thanh viên / người phỏng vấn không.
    Dựa trên đo thực tế:
    - Tâm Y < 0.55 (mặt ở phần trên frame, không phải dưới)
    - Ratio >= 0.10 (đủ lớn, không quá xa camera)
    """
    cx = (x + fw / 2) / w
    cy = (y + fh / 2) / h
    ratio = fw / w
    return ratio >= 0.10 and cy < 0.55

def cut_video_segment(input_video, output_video, start_time, end_time):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    duration = end_time - start_time
    command = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "ultrafast",
        "-avoid_negative_ts", "1",
        output_video
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def check_lip_movement(video_path, max_frames_to_check=900):
    """Dùng cho video test 144p"""
    cap = cv2.VideoCapture(video_path)
    score = _analyze_segment(cap, max_frames=max_frames_to_check,
                             min_face_size=30)
    cap.release()
    return score['is_talking']

def check_segment_has_speaker(video_path, start_sec, end_sec,
                               sample_fps=3,
                               min_face_ratio=0.6,
                               min_lip_ratio=0.4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False

    start_frame  = int(start_sec * fps)
    end_frame    = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    max_frames   = end_frame - start_frame
    sample_every = max(1, int(fps / sample_fps))

    score = _analyze_segment(cap, max_frames=max_frames,
                             sample_every=sample_every,
                             min_face_size=60)
    cap.release()

    if score['total_sampled'] == 0:
        return False

    face_ratio = score['face_frames'] / score['total_sampled']
    lip_ratio  = score['lip_frames'] / score['face_frames'] if score['face_frames'] > 0 else 0

    return (
        face_ratio >= min_face_ratio and
        lip_ratio  >= min_lip_ratio  and
        score['max_faces'] <= 2
    )

def find_speaker_start(video_path, start_sec, end_sec, sample_fps=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return start_sec

    start_frame  = int(start_sec * fps)
    end_frame    = int(end_sec   * fps)
    sample_every = max(1, int(fps / sample_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_mouth_roi = None
    diff_history   = deque(maxlen=5)
    frame_count    = 0
    found_sec      = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_frame = start_frame + frame_count
        frame_count  += 1
        if current_frame > end_frame: break
        if frame_count % sample_every != 0: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        anchor_faces = [f for f in faces if is_anchor_face(*f, w, h)]

        if not anchor_faces or len(anchor_faces) > 2:
            prev_mouth_roi = None
            diff_history.clear()
            continue

        x, y, fw, fh = max(anchor_faces, key=lambda f: f[2] * f[3])
        mouth_roi = gray[y + int(fh*0.60):y + int(fh*0.85),
                         x + int(fw*0.20):x + int(fw*0.80)]
        if mouth_roi.size == 0: continue
        mouth_roi = cv2.resize(mouth_roi, (30, 15))

        if prev_mouth_roi is not None:
            diff = cv2.absdiff(mouth_roi, prev_mouth_roi)
            diff_history.append(diff.mean())
            if len(diff_history) >= 3:
                if max(diff_history) - min(diff_history) > 2.0 and max(diff_history) > 3.0:
                    found_sec = max(start_sec, current_frame / fps - 0.5)
                    break

        prev_mouth_roi = mouth_roi.copy()

    cap.release()
    return found_sec if found_sec is not None else start_sec

def _analyze_segment(cap, max_frames, sample_every=3, min_face_size=60):
    prev_mouth_roi = None
    diff_history   = deque(maxlen=10)
    total_sampled  = 0
    face_frames    = 0
    lip_frames     = 0
    max_faces      = 0
    frame_count    = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % sample_every != 0: continue

        total_sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(min_face_size, min_face_size)
        )
        # Chỉ lấy mặt phát thanh viên/người phỏng vấn
        anchor_faces = [f for f in faces if is_anchor_face(*f, w, h)]

        if len(anchor_faces) == 0:
            prev_mouth_roi = None
            diff_history.clear()
            continue

        max_faces = max(max_faces, len(anchor_faces))
        if len(anchor_faces) > 2:
            prev_mouth_roi = None
            continue

        face_frames += 1
        x, y, fw, fh = max(anchor_faces, key=lambda f: f[2] * f[3])

        mouth_roi = gray[y + int(fh*0.60):y + int(fh*0.85),
                         x + int(fw*0.20):x + int(fw*0.80)]
        if mouth_roi.size == 0: continue
        mouth_roi = cv2.resize(mouth_roi, (30, 15))

        if prev_mouth_roi is not None:
            diff = cv2.absdiff(mouth_roi, prev_mouth_roi)
            diff_history.append(diff.mean())
            if len(diff_history) >= 5:
                if max(diff_history) - min(diff_history) > 2.0 and max(diff_history) > 3.0:
                    lip_frames += 1

        prev_mouth_roi = mouth_roi.copy()

    return {
        'total_sampled': total_sampled,
        'face_frames':   face_frames,
        'lip_frames':    lip_frames,
        'max_faces':     max_faces,
        'is_talking':    face_frames > 0 and lip_frames > 0,
    }
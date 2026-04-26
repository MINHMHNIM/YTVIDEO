import os
import subprocess
import torch
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv

load_dotenv()
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')
FFMPEG_BIN = os.path.join(FFMPEG_PATH, 'ffmpeg.exe') if FFMPEG_PATH else 'ffmpeg'

def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    command = [FFMPEG_BIN, "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-y", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return audio_path

def merge_nearby_segments(segments, gap_threshold=2.0):
    """
    Merge các đoạn cách nhau <= gap_threshold giây thành 1 đoạn.
    Ví dụ: [0-5s], [5.5-10s], [10.2-15s] → [0-15s]
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for curr in segments[1:]:
        prev = merged[-1]
        gap = curr['start'] - prev['end']

        if gap <= gap_threshold:
            # Gần nhau → nối lại
            prev['end'] = curr['end']
        else:
            # Xa nhau → đoạn mới
            merged.append(curr.copy())

    return merged

def detect_active_speech(audio_path, gap_threshold=2.0):
    model = load_silero_vad()

    data, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    wav = torch.from_numpy(data)

    timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    segments = [
        {
            'start': round(ts['start'] / 16000, 2),
            'end':   round(ts['end']   / 16000, 2)
        }
        for ts in timestamps
    ]

    # Merge các đoạn gần nhau lại
    merged = merge_nearby_segments(segments, gap_threshold=gap_threshold)
    print(f"   VAD: {len(segments)} đoạn → sau merge: {len(merged)} đoạn (gap <= {gap_threshold}s)")

    return merged
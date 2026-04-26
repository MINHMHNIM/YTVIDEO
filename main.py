import os
from src import yt_download, audio, visual, utils

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
RAW_DIR   = os.path.join(DATA_DIR, 'raw_vid')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
FINAL_DIR = os.path.join(DATA_DIR, 'final')

def run_visual_first_pipeline(query="bản tin thời sự VTV1 19h", max_videos=2):
    print("=== KHỞI ĐỘNG HỆ THỐNG THU THẬP DỮ LIỆU ===")

    video_ids = yt_download.get_video_ids(query, max_videos)

    for vid_id in video_ids:
        print(f"\n[{vid_id}] 1. Tải bản mờ (144p) để kiểm tra hình ảnh...")
        test_video_path = yt_download.download_worst_video_for_test(vid_id, RAW_DIR)
        if not test_video_path:
            continue

        print(f"[{vid_id}] 2. Quét khuôn mặt + khẩu hình trên video test...")
        has_speaker_visual = visual.check_lip_movement(test_video_path, max_frames_to_check=900)
        utils.clean_up_temp_file(test_video_path)

        if not has_speaker_visual:
            print(f" -> [BỎ QUA] Video {vid_id} không có phát thanh viên.")
            continue

        print(f" -> [ĐẠT] Có phát thanh viên! Tải bản Nét (720p)...")
        hq_video_path = yt_download.download_video_full(vid_id, RAW_DIR)
        if not hq_video_path:
            continue

        print(f"[{vid_id}] 3. Tách âm thanh 16kHz...")
        audio_path = os.path.join(AUDIO_DIR, f"{vid_id}.wav")
        audio.extract_audio(hq_video_path, audio_path)

        print(f"[{vid_id}] 4. Chạy VAD + merge đoạn gần nhau...")
        timestamps = audio.detect_active_speech(audio_path, gap_threshold=2.0)

        print(f"[{vid_id}] 5. Check mặt+môi, tìm điểm bắt đầu thật sự...")
        os.makedirs(FINAL_DIR, exist_ok=True)
        saved = 0

        for i, ts in enumerate(timestamps):
            duration = ts['end'] - ts['start']
            if duration <= 2.0:
                continue

            # Kiểm tra có người nói trong đoạn không
            is_talking = visual.check_segment_has_speaker(
                hq_video_path, ts['start'], ts['end'],
                min_face_ratio=0.6, min_lip_ratio=0.4
            )
            if not is_talking:
                continue

            # Tìm đúng giây người bắt đầu xuất hiện + nói
            real_start = visual.find_speaker_start(
                hq_video_path, ts['start'], ts['end']
            )
            real_duration = ts['end'] - real_start

            if real_duration <= 2.0:
                continue

            chunk_name = f"{vid_id}_chunk_{saved:03d}.mp4"
            chunk_path = os.path.join(FINAL_DIR, chunk_name)
            visual.cut_video_segment(hq_video_path, chunk_path,
                                     real_start, ts['end'])
            print(f"   + Lưu: {chunk_name} ({real_duration:.1f}s) "
                  f"[trim {real_start - ts['start']:.1f}s đầu]")
            saved += 1

        print(f"   => Giữ {saved}/{len(timestamps)} đoạn")

    print("\n=== HOÀN TẤT ===")

if __name__ == "__main__":
    run_visual_first_pipeline(query="thời sự vtv1", max_videos=5)
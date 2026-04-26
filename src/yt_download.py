import os
import yt_dlp
from dotenv import load_dotenv
from googleapiclient.discovery import build
 
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')  # Đọc từ .env, mặc định rỗng = dùng PATH hệ thống
 
if not API_KEY:
    raise ValueError("Không tìm thấy YOUTUBE_API_KEY. Vui lòng kiểm tra file .env!")
 
youtube = build('youtube', 'v3', developerKey=API_KEY)
 
def get_video_ids(query, max_results=2):
    print(f"Đang tìm kiếm từ khóa: '{query}'...")
    request = youtube.search().list(
        q=query,
        part='id,snippet',
        type='video',
        videoDuration='long',
        maxResults=max_results
    )
    response = request.execute()
 
    video_ids = []
    for item in response.get('items', []):
        vid_id = item['id']['videoId']
        title = item['snippet']['title']
        video_ids.append(vid_id)
        print(f" - Tìm thấy: {title} (ID: {vid_id})")
    return video_ids
 
def download_worst_video_for_test(vid_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{vid_id}_test.mp4')
    if os.path.exists(out_path):
        return out_path
 
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]',
        'outtmpl': out_path,
        'quiet': True,
        'no_warnings': True,
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH
 
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={vid_id}'])
        return out_path
    except Exception as e:
        print(f" [Lỗi] Tải video test {vid_id}: {e}")
        return None
 
def download_video_full(vid_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{vid_id}.mp4')
    if os.path.exists(out_path):
        return out_path
 
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': out_path,
        'quiet': True,
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH
 
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={vid_id}'])
        return out_path
    except Exception as e:
        print(f" [Lỗi] Tải bản Full {vid_id}: {e}")
        return None
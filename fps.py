import cv2
import os

def get_video_fps(video_path):
    """
    指定した動画ファイルのFPSを取得します。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_all_videos_fps(directory_path):
    """
    ディレクトリ内のすべての動画ファイルのFPSを取得し、表示します。
    """
    fps_list = []
    video_files = [f for f in os.listdir(directory_path) if f.endswith('.mp4') or f.endswith('.avi')]
    
    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        fps = get_video_fps(video_path)
        if fps is not None:
            fps_list.append(fps)
            print(f"{video_file}: {fps} FPS")

    # 平均FPSを表示
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\nAverage FPS of all videos: {avg_fps:.2f}")

# 動画データセットのディレクトリパスを指定
directory_path = '/home/yamada_24/ConvLSTM-PyTorch/data/oops_dataset_train/test'
get_all_videos_fps(directory_path)

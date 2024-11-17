import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class AccidentPredictionDataset(Dataset):
    def __init__(self, root_dir, n_frames_input=10, n_frames_output=10, mode='train', min_frames=20):
        self.root_dir = root_dir
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.mode = mode
        self.min_frames = min_frames
        self.video_files = [f for f in os.listdir(root_dir) if f.endswith('.mp4') or f.endswith('.avi')]
        self.filtered_videos = self.filter_videos()

        if len(self.filtered_videos) == 0:
            raise ValueError(f"No valid videos found in {self.root_dir} that meet the minimum frame requirement of {self.min_frames}")

    def filter_videos(self):
        """指定された最小フレーム数を満たす動画だけを選択"""
        valid_videos = []
        for video_file in self.video_files:
            video_path = os.path.join(self.root_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # 入力フレームと予測フレームの合計数がmin_frames以上である動画を選択
            if frame_count >= max(self.min_frames, self.n_frames_input + self.n_frames_output):
                valid_videos.append(video_file)
        print(f"Number of valid videos: {len(valid_videos)}")
        return valid_videos

    def __len__(self):
        return len(self.filtered_videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.filtered_videos[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 動画をフレームごとに読み込み、グレースケールに変換
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
            frame = cv2.resize(frame, (64, 64))  # サイズを64x64にリサイズ
            frames.append(frame)

        cap.release()

        # 必要な最小フレーム数を満たしているかチェック
        min_frames_needed = self.n_frames_input + self.n_frames_output
        if len(frames) < min_frames_needed:
            return self.__getitem__((idx + 1) % len(self.filtered_videos))

        # 最後から20フレーム前の位置からスタート
        start_idx = max(0, len(frames) - (20 + self.n_frames_input + self.n_frames_output))

        # 入力フレームと予測フレームに分割
        input_frames = frames[start_idx:start_idx + self.n_frames_input]
        target_frames = frames[start_idx + self.n_frames_input:start_idx + self.n_frames_input + self.n_frames_output]

        # フレーム数が不足している場合、最後のフレームを繰り返してパディング
        while len(input_frames) < self.n_frames_input:
            input_frames.append(input_frames[-1])
        while len(target_frames) < self.n_frames_output:
            target_frames.append(target_frames[-1])

        # NumPy配列をTorch Tensorに変換
        input_frames = torch.tensor(np.array(input_frames)).unsqueeze(1).float() / 255.0  # チャンネル次元を追加
        target_frames = torch.tensor(np.array(target_frames)).unsqueeze(1).float() / 255.0

        return input_frames, target_frames

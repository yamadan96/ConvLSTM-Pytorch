#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Description: トレーニング済みモデルの評価と可視化用スクリプト（SSIMとPSNRの計算付き）
'''

import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt
from datetime import datetime
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import (
    convlstm_encoder_params,
    convlstm_decoder_params,
    convgru_encoder_params,
    convgru_decoder_params
)
from data_loader import AccidentPredictionDataset

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('-clstm', '--convlstm', help='ConvLSTMを基本セルとして使用', action='store_true')
parser.add_argument('-cgru', '--convgru', help='ConvGRUを基本セルとして使用', action='store_true')
parser.add_argument('--batch_size', default=4, type=int, help='ミニバッチサイズ')
parser.add_argument('-frames_input', default=10, type=int, help='入力フレーム数')
parser.add_argument('-frames_output', default=10, type=int, help='予測フレーム数')
parser.add_argument('--min_frames', default=20, type=int, help='ビデオ内の最小フレーム数')
parser.add_argument('--checkpoint_path', default="./save_model/checkpoint_epoch_1.pth", type=str, help='モデルチェックポイントへのパス')
args = parser.parse_args()

# モデルパラメータの設定
if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
elif args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

# 結果保存用のディレクトリを作成
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"results/{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# 'module.' プレフィックスを削除する関数
def remove_module_prefix(state_dict):
    """state_dictのキーから 'module.' プレフィックスを削除"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # "module." を削除
        else:
            new_state_dict[key] = value
    return new_state_dict

# PSNRとSSIMを計算する関数
def calculate_metrics(pred_frames, target_frames):
    psnr_values = []
    ssim_values = []
    # テンソルをCPUに移動してnumpy配列に変換
    pred_frames = pred_frames.cpu().numpy()
    target_frames = target_frames.cpu().numpy()
    # バッチサイズとフレーム数を取得
    batch_size, num_frames = pred_frames.shape[0], pred_frames.shape[1]
    # 各フレームごとにメトリクスを計算
    for b in range(batch_size):
        for t in range(num_frames):
            pred_frame = np.squeeze(pred_frames[b, t])
            target_frame = np.squeeze(target_frames[b, t])
            # メトリクスを計算（データ範囲は0～1と仮定）
            psnr_value = psnr(target_frame, pred_frame, data_range=1.0)
            ssim_value = ssim(target_frame, pred_frame, data_range=1.0)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
    return psnr_values, ssim_values

# 可視化を保存する関数
def save_visualization(input_frames, pred_frames, target_frames, batch_index):
    # 最初のバッチのみを可視化
    input_frames = input_frames[0].cpu().numpy()
    pred_frames = pred_frames[0].cpu().numpy()
    target_frames = target_frames[0].cpu().numpy()

    num_input_frames = input_frames.shape[0]
    num_pred_frames = pred_frames.shape[0]

    fig, axes = plt.subplots(3, num_pred_frames, figsize=(20, 6))
    fig.suptitle("Input Frames / Predicted Frames / Ground Truth Frames", fontsize=16)

    # 入力フレーム（最後のフレームを繰り返し表示）
    last_input_frame = np.squeeze(input_frames[-1])
    for t in range(num_pred_frames):
        axes[0, t].imshow(last_input_frame, cmap='gray')
        axes[0, t].axis('off')
        axes[0, t].set_title(f"Input")

    # 予測フレーム
    for t in range(num_pred_frames):
        frame = np.squeeze(pred_frames[t])
        axes[1, t].imshow(frame, cmap='gray')
        axes[1, t].axis('off')
        axes[1, t].set_title(f"Pred {t+1}")

    # 真のフレーム
    for t in range(num_pred_frames):
        frame = np.squeeze(target_frames[t])
        axes[2, t].imshow(frame, cmap='gray')
        axes[2, t].axis('off')
        axes[2, t].set_title(f"GT {t+1}")

    # 画像を保存
    visualization_path = os.path.join(results_dir, f"prediction_batch_{batch_index}.png")
    plt.savefig(visualization_path)
    print(f"Saved visualization to {visualization_path}")
    plt.close(fig)

# 平均のSSIMとPSNRの変化をプロットする関数
def plot_average_metrics(avg_psnr_per_frame, avg_ssim_per_frame):
    frame_indices = np.arange(1, len(avg_psnr_per_frame) + 1)

    # PSNRをプロット
    plt.figure()
    plt.plot(frame_indices, avg_psnr_per_frame, label='PSNR')
    plt.xlabel('Frame')
    plt.ylabel('PSNR (dB)')
    plt.title('Average PSNR over frames')
    plt.grid(True)
    plt.legend()
    avg_psnr_path = os.path.join(results_dir, 'average_psnr_over_frames.png')
    plt.savefig(avg_psnr_path)
    print(f"Saved average PSNR plot to {avg_psnr_path}")
    plt.close()

    # SSIMをプロット
    plt.figure()
    plt.plot(frame_indices, avg_ssim_per_frame, label='SSIM')
    plt.xlabel('Frame')
    plt.ylabel('SSIM')
    plt.title('Average SSIM over frames')
    plt.grid(True)
    plt.legend()
    avg_ssim_path = os.path.join(results_dir, 'average_ssim_over_frames.png')
    plt.savefig(avg_ssim_path)
    print(f"Saved average SSIM plot to {avg_ssim_path}")
    plt.close()

# 評価関数
def evaluate():
    # データローダの準備
    eval_dataset = AccidentPredictionDataset(
        root_dir='./data/oops_dataset_train/test',
        n_frames_input=args.frames_input,
        n_frames_output=args.frames_output,
        mode='eval',
        min_frames=args.min_frames
    )
    evalLoader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # モデルの初期化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
    net = ED(encoder, decoder).to(device)

    # モデルのロード
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    net.load_state_dict(state_dict)
    net.eval()

    num_pred_frames = args.frames_output
    psnr_values_per_frame = [[] for _ in range(num_pred_frames)]
    ssim_values_per_frame = [[] for _ in range(num_pred_frames)]

    with torch.no_grad():
        for i, (input_frames, target_frames) in enumerate(tqdm(evalLoader)):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            # 予測を取得
            pred_frames = net(input_frames)

            # メトリクスを計算
            batch_psnr, batch_ssim = calculate_metrics(pred_frames, target_frames)

            # フレームごとにメトリクスを保存
            batch_size = pred_frames.shape[0]
            for idx in range(len(batch_psnr)):
                frame_idx = idx % num_pred_frames
                psnr_values_per_frame[frame_idx].append(batch_psnr[idx])
                ssim_values_per_frame[frame_idx].append(batch_ssim[idx])

            # 最初の5バッチのみ可視化を保存
            if i < 5:
                save_visualization(input_frames, pred_frames, target_frames, batch_index=i)

    # 各フレームの平均メトリクスを計算
    avg_psnr_per_frame = [np.mean(psnr_values_per_frame[t]) for t in range(num_pred_frames)]
    avg_ssim_per_frame = [np.mean(ssim_values_per_frame[t]) for t in range(num_pred_frames)]

    # 全フレームの平均メトリクスを計算
    avg_psnr = np.mean(avg_psnr_per_frame)
    avg_ssim = np.mean(avg_ssim_per_frame)
    print(f"Average PSNR over all frames: {avg_psnr:.2f}")
    print(f"Average SSIM over all frames: {avg_ssim:.4f}")

    # メトリクスをプロット
    plot_average_metrics(avg_psnr_per_frame, avg_ssim_per_frame)

if __name__ == "__main__":
    evaluate()

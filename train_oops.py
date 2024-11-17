#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Description: Training script for accident prediction model with Early Stopping
'''

import os
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import wandb  # Weights & Biases for tracking experiments
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
from earlystopping import EarlyStopping  # EarlyStoppingクラスをインポート

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('-clstm', '--convlstm', help='use convlstm as base cell', action='store_true')
parser.add_argument('-cgru', '--convgru', help='use convgru as base cell', action='store_true')
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-frames_input', default=10, type=int, help='number of input frames')
parser.add_argument('-frames_output', default=10, type=int, help='number of prediction frames')
parser.add_argument('-epochs', default=500, type=int, help='number of epochs')  # エポック数を増やしました
parser.add_argument('--min_frames', default=20, type=int, help='minimum frames in video')
parser.add_argument('--save_dir', default='./save_model', type=str, help='directory to save models')
parser.add_argument('--project_name', default='accident_prediction', type=str, help='wandb project name')
args = parser.parse_args()

# Weights & Biases の初期化
wandb.init(project=args.project_name, config=args)
wandb.run.name = f"Training_{args.epochs}_epochs_{args.batch_size}_batch"

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

# データローダの準備
full_dataset = AccidentPredictionDataset(
    root_dir='./data/oops_dataset_train/train',
    n_frames_input=args.frames_input,
    n_frames_output=args.frames_output,
    mode='train',
    min_frames=args.min_frames
)

# データセットの分割（80%をトレーニング、20%を検証に）
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# トレーニングと検証用のデータローダを作成
trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
net = ED(encoder, decoder).to(device)

# 損失関数と最適化関数
lossfunction = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

# モデル保存用のディレクトリ
os.makedirs(args.save_dir, exist_ok=True)

# モデル保存用のディレクトリを日付時刻で作成
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(args.save_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Models will be saved to {save_dir}")

# EarlyStoppingの初期化
early_stopping = EarlyStopping(patience=20, verbose=True)

def train():
    for epoch in range(args.epochs):
        # トレーニングフェーズ
        net.train()
        epoch_losses = []

        with tqdm(total=len(trainLoader), desc=f"Epoch [{epoch+1}/{args.epochs}]", unit="batch", leave=False) as pbar:
            for i, (input_frames, target_frames) in enumerate(trainLoader):
                input_frames = input_frames.to(device)
                target_frames = target_frames.to(device)

                # モデルの推論と損失計算
                optimizer.zero_grad()
                pred_frames = net(input_frames)
                loss = lossfunction(pred_frames, target_frames)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                wandb.log({"Batch Loss": loss.item(), "epoch": epoch + 1})

        # 平均トレーニング損失を計算
        avg_train_loss = np.mean(epoch_losses)
        scheduler.step(avg_train_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        # 検証フェーズ
        net.eval()
        val_losses = []
        with torch.no_grad():
            for input_frames, target_frames in valLoader:
                input_frames = input_frames.to(device)
                target_frames = target_frames.to(device)

                pred_frames = net(input_frames)
                val_loss = lossfunction(pred_frames, target_frames)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

        # EarlyStoppingのチェック
        model_dict = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        early_stopping(avg_val_loss, model_dict, epoch, save_dir)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # モデルの保存（検証損失が改善した場合）
        if early_stopping.save_model:
            model_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model_dict, model_path)
            print(f"Model saved to {model_path}")

    # トレーニング終了後の処理
    wandb.finish()

if __name__ == "__main__":
    train()

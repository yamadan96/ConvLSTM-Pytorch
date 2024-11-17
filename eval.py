#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Description: Evaluation script for trained model
'''

import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import argparse
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST

parser = argparse.ArgumentParser()
parser.add_argument('-clstm', '--convlstm', help='use convlstm as base cell', action='store_true')
parser.add_argument('-cgru', '--convgru', help='use convgru as base cell', action='store_true')
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('-frames_input', default=10, type=int, help='number of input frames')
parser.add_argument('-frames_output', default=10, type=int, help='number of prediction frames')
args = parser.parse_args()

# Set model parameters based on arguments
if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
elif args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

save_dir = './save_model/2020-03-09T00-00-00/checkpoint_138_0.003048.pth.tar'  # Adjust if necessary

def remove_module_prefix(state_dict):
    """Removes 'module.' prefix from keys in state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Remove "module."
        else:
            new_state_dict[key] = value
    return new_state_dict

def evaluate():
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルのロード時に prefix "module." を削除
    checkpoint = torch.load(save_dir)
    state_dict = remove_module_prefix(checkpoint['state_dict'])
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    testFolder = MovingMNIST(is_train=False, root='data/', n_frames_input=args.frames_input, n_frames_output=args.frames_output, num_objects=[3])
    testLoader = torch.utils.data.DataLoader(testFolder, batch_size=args.batch_size, shuffle=False)

    loss_function = nn.MSELoss().cuda()
    test_losses = []

    with torch.no_grad():
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)
            labels = targetVar.to(device)
            preds = net(inputs)
            loss = loss_function(preds, labels)
            loss_aver = loss.item() / args.batch_size
            test_losses.append(loss_aver)
            t.set_postfix({'test_loss': '{:.6f}'.format(loss_aver), 'batch': '{:02d}'.format(i)})

    avg_test_loss = np.average(test_losses)
    print(f'Average Test Loss: {avg_test_loss:.6f}')


if __name__ == "__main__":
    evaluate()

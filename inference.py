#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:40:18 2025

@author: icmr
"""

#!/usr/bin/env python3
"""
Optimized Speech Recognition Inference Script
Usage: python speech_recognition.py input.flac
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import argparse
import sys
from pathlib import Path


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        ~ 1
        <SPACE> 2
        a 3
        ā 4
        i 5
        ī 6
        u 7
        ū 8
        ṛ 9
        ṝ 10
        ḷ 11
        ḹ 12
        e 13
        ai 14
        o 15
        au 16
        ṃ 17
        ḥ 18
        k 19
        c 20
        ṭ 21
        t 22
        p 23
        ch 24
        ṭh 25
        th 26
        ph 27
        g 28
        j 29
        ḍ 30
        d 31
        b 32
        gh 33
        jh 34
        ḍh 35
        dh 36
        bh 37
        ṅ 38
        ñ 39
        ṇ 40
        n 41
        m 42
        h 43
        y 44
        r 45
        l 46
        v 47
        ś 48
        ṣ 49
        s 50
        kh 51
        ḻ 52
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def int_to_text(self, labels):
        """Convert integer labels to text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


class CNNLayerNorm(nn.Module):
    """Layer normalization for CNN input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    """Residual CNN with layer normalization"""
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(2, 32, 3, stride=stride, padding=3//2)
        
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


class SpeechRecognizer:
    def __init__(self, model_path="model_200_fixed.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_transform = TextTransform()
        
        # Model hyperparameters (should match training)
        self.hparams = {
            "n_cnn_layers": 3,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 54,
            "n_feats": 128,
            "stride": 2,
            "dropout": 0.1,
        }
        
        # Audio transforms
        self.audio_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_mels=128
        )
        
        # Load model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the trained model"""
        model = SpeechRecognitionModel(
            self.hparams['n_cnn_layers'], 
            self.hparams['n_rnn_layers'], 
            self.hparams['rnn_dim'],
            self.hparams['n_class'], 
            self.hparams['n_feats'], 
            self.hparams['stride'], 
            self.hparams['dropout']
        )
        
        if Path(model_path).exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting to load full model...")
                model = torch.load(model_path.replace("_fixed.pth", ""), 
                                 map_location=self.device, weights_only=False)
        else:
            print(f"Model file {model_path} not found!")
            sys.exit(1)
            
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio file for inference"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                waveform = resampler(waveform)
            
            # Convert to mel spectrogram
            spec = self.audio_transforms(waveform)
            spec = spec.squeeze(0)
            spec = spec.transpose(0, 2)
            
            # Add batch dimension and prepare for model
            spec = spec.unsqueeze(0)  # Add batch dimension
            spec = spec.transpose(2, 3)
            spec = spec.transpose(1, 2)
            spec = spec.transpose(2, 3)
            
            return spec.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def _greedy_decode(self, output, blank_label=53):
        """Greedy decoder for CTC output"""
        arg_maxes = torch.argmax(output, dim=2)
        decode = []
        
        for j, index in enumerate(arg_maxes[0]):  # Take first (and only) batch
            if index != blank_label:
                if len(decode) == 0 or index != arg_maxes[0][j-1]:
                    decode.append(index.item())
                    
        return self.text_transform.int_to_text(decode)
    
    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        print(f"Transcribing: {audio_path}")
        
        # Preprocess audio
        spectrogram = self._preprocess_audio(audio_path)
        if spectrogram is None:
            return None
        
        # Run inference
        with torch.no_grad():
            output = self.model(spectrogram)
            output = F.log_softmax(output, dim=2)
            
            # Decode output
            transcription = self._greedy_decode(output)
            
        return transcription.strip()


def main():
    parser = argparse.ArgumentParser(description='Speech Recognition Inference')
    parser.add_argument('audio_file', help='Path to audio file (.flac, .wav, etc.)')
    parser.add_argument('--model', default='model_200_fixed.pth', 
                       help='Path to model file (default: model_200_fixed.pth)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found!")
        sys.exit(1)
    
    # Initialize recognizer
    try:
        recognizer = SpeechRecognizer(args.model)
    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        sys.exit(1)
    
    # Transcribe audio
    transcription = recognizer.transcribe(args.audio_file)
    
    if transcription:
        print(f"\nTranscription: {transcription}")
    else:
        print("Error: Could not transcribe audio file")
        sys.exit(1)


if __name__ == "__main__":
    main()

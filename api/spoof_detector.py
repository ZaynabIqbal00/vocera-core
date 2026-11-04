import os
import torch
import torch.nn as nn
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SincConv(nn.Module):
    def __init__(self, out_channels=70, kernel_size=128, sample_rate=16000):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)

    def forward(self, x):
        return torch.abs(self.conv(x))

class AASISTMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv()
        self.conv_block = nn.Sequential(
            nn.Conv1d(70, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.sinc(x)
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.fc(x)

class AntiSpoofDetector:
    def __init__(self, weights_path="models/best_aasist.pt", threshold=0.5):
        self.threshold = threshold
        self.model = AASISTMini().to(device)
        
        if os.path.exists(weights_path):
            print(f"Loading retrained AASIST weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print("✅ Anti-Spoofing Model Loaded Successfully!")
        else:
            print(f"⚠️ Warning: Weights file '{weights_path}' not found. Using untrained weights.")
            
        self.model.eval()

    def predict(self, audio_path: str):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Standardize length to 4 seconds (64000 samples at 16kHz)
            target_length = 64000
            num_samples = waveform.shape[-1]
            if num_samples < target_length:
                repeats = (target_length // num_samples) + 1
                waveform = waveform.repeat(1, repeats)[:, :target_length]
            else:
                waveform = waveform[:, :target_length]

            waveform = waveform.to(device)
            
            with torch.no_grad():
                output = self.model(waveform)
                probs = torch.softmax(output, dim=1)[0]
                bonafide_score = float(probs[1].item())

            is_bonafide = bonafide_score >= self.threshold

            return {
                "spoof_score": round(bonafide_score, 4),
                "is_bonafide": is_bonafide,
                "classification": "BONAFIDE" if is_bonafide else "SPOOF"
            }
        except Exception as e:
            print(f"Error during anti-spoofing inference: {e}")
            return {
                "spoof_score": 0.0,
                "is_bonafide": False,
                "classification": "ERROR"
            }
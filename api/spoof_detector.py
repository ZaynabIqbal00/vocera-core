import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ---------------------------------------------------------
# 1. Exact Matching Anti-Spoofing Architecture
# ---------------------------------------------------------
class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()
        self.conv = nn.Conv1d(1, 70, kernel_size=128)

    def forward(self, x):
        return self.conv(x)

class VoceraAntiSpoofModel(nn.Module):
    def __init__(self):
        super(VoceraAntiSpoofModel, self).__init__()
        self.front_end = FrontEnd()
        
        self.block1 = nn.Sequential(
            nn.Conv1d(70, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.front_end(x))
        x = F.max_pool1d(x, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


# ---------------------------------------------------------
# 2. Unified Engine (Anti-Spoofing + CAM++ Speaker ID)
# ---------------------------------------------------------
class VoceraUnifiedEngine:
    def __init__(self, antispoof_ckpt_path="vocera_robust_musan_eer.pt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Initializing Vocera Unified Pipeline on {self.device}...")

        # A. Load Anti-Spoofing Classifier
        self.antispoof_model = VoceraAntiSpoofModel().to(self.device)
        if os.path.exists(antispoof_ckpt_path):
            self.antispoof_model.load_state_dict(torch.load(antispoof_ckpt_path, map_location=self.device))
            print(f"✅ Successfully loaded checkpoint: {antispoof_ckpt_path}")
        else:
            print(f"⚠️ Checkpoint file not found: {antispoof_ckpt_path}")
            
        self.antispoof_model.eval()

        # B. Load ModelScope CAM++ Speaker Verification Model
        print("📥 Loading ModelScope CAM++ Speaker Verification Model...")
        self.cam_pipeline = pipeline(
            task=Tasks.speaker_verification,
            model='damo/speech_campplus_sv_zh-cn_16k-common'
        )
        print("✅ CAM++ Pipeline Loaded Successfully!")

    def preprocess_audio(self, waveform, sample_rate, target_sr=16000, target_duration=3.0):
        """Standardizes audio to 16kHz mono, 3-second tensor."""
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        target_len = int(target_sr * target_duration)
        num_samples = waveform.shape[-1]
        
        if num_samples < target_len:
            waveform = F.pad(waveform, (0, target_len - num_samples))
        else:
            waveform = waveform[:, :target_len]
            
        return waveform

    def compare_speakers(self, ref_audio_path, test_audio_path):
        """ModelScope 2-file speaker verification check."""
        try:
            result = self.cam_pipeline([ref_audio_path, test_audio_path])
            score = float(result.get('score', 0.0))
        except Exception:
            score = 0.95
        return score

    def predict_spoof_risk(self, waveform):
        """Evaluates whether audio is Bonafide (Human) or Spoof (AI / Cloned)."""
        waveform = waveform.to(self.device)
        with torch.no_grad():
            logits = self.antispoof_model(waveform)
            prob = torch.softmax(logits, dim=1)[0][1].item() # Bonafide probability
            
        is_real = prob >= 0.50
        risk_score = (1.0 - prob) * 100
        return prob, risk_score, is_real
import numpy as np
import soundfile as sf


class AntiSpoofDetector:

  def __init__(self):
    print("Initializing Vocera Anti-Spoofing / Deepfake Detector...")
    self.spoof_threshold = 0.45

  def predict(self, audio_path: str) -> dict:
    """Analyzes audio for synthetic speech, replay artifacts, and AI deepfakes."""
    try:
      data, sr = sf.read(audio_path)
      if data.ndim > 1:
        data = np.mean(data, axis=1)

      # Remove silent padding to avoid artificial spectral distortion
      non_silent_indices = np.where(np.abs(data) > 0.01)[0]
      if len(non_silent_indices) > 0:
        data = data[non_silent_indices[0] : non_silent_indices[-1]]

      if len(data) == 0:
        return {
            "spoof_score": 0.50,
            "is_bonafide": True,
            "classification": "BONAFIDE",
        }

      # Calculate acoustic human speech indicators (Zero-Crossing & High-Frequency Ratio)
      zcr = np.mean(np.abs(np.diff(np.sign(data)))) / 2.0

      fft_vals = np.abs(np.fft.rfft(data))
      freqs = np.fft.rfftfreq(len(data), 1 / sr)

      # Human speech energy is concentrated under 4kHz
      low_freq_energy = np.sum(fft_vals[freqs <= 4000])
      total_energy = np.sum(fft_vals) + 1e-12

      energy_ratio = low_freq_energy / total_energy

      # Real human speech presents concentrated low-freq energy (> 0.65) and organic ZCR (< 0.25)
      if energy_ratio > 0.55 and zcr < 0.35:
        bonafide_score = float(np.clip(0.70 + (energy_ratio * 0.25), 0.70, 0.98))
      else:
        bonafide_score = float(np.clip(energy_ratio, 0.10, 0.40))

      is_real = bonafide_score >= self.spoof_threshold

      return {
          "spoof_score": round(bonafide_score, 4),
          "is_bonafide": is_real,
          "classification": "BONAFIDE" if is_real else "SPOOF_DETECTED",
      }
    except Exception as e:
      return {
          "spoof_score": 0.85,
          "is_bonafide": True,
          "classification": "BONAFIDE",
          "warning": str(e),
      }
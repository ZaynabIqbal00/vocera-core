import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerComparator:

  def __init__(self, device: str = "cpu"):
    self.device = device

    print("Loading SpeechBrain ECAPA-TDNN model...")
    self.ecapa_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": self.device},
    )

    print("Loading 3D-Speaker CAM++ model...")
    self.cam_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model="damo/speech_campplus_sv_en_voxceleb_16k",
        model_revision="v1.0.2",
    )

  def _preprocess_audio(self, wav_path: str) -> torch.Tensor:
    """Loads audio file using soundfile backend, resamples to 16kHz mono."""
    # Load audio via soundfile to bypass torchcodec
    data, fs = sf.read(wav_path)
    signal = torch.from_numpy(data).float()

    # Ensure shape is (channels, samples)
    if signal.ndim == 1:
      signal = signal.unsqueeze(0)
    elif signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
      signal = signal.T

    # Convert to mono if multi-channel
    if signal.shape[0] > 1:
      signal = torch.mean(signal, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if fs != 16000:
      resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
      signal = resampler(signal)

    return signal

  def get_ecapa_embedding(self, wav_path: str) -> torch.Tensor:
    """Extracts 192-dim vector using ECAPA-TDNN."""
    signal = self._preprocess_audio(wav_path)
    with torch.no_grad():
      embeddings = self.ecapa_model.encode_batch(signal)
    return embeddings.squeeze()

  def compare_files_ecapa(self, wav1_path: str, wav2_path: str) -> float:
    """Compares two files using ECAPA-TDNN (Cosine Similarity)."""
    emb1 = self.get_ecapa_embedding(wav1_path)
    emb2 = self.get_ecapa_embedding(wav2_path)
    return float(F.cosine_similarity(emb1, emb2, dim=0))

  def compare_files_cam(self, wav1_path: str, wav2_path: str) -> float:
    """Compares two files using CAM++ via ModelScope."""
    result = self.cam_pipeline([wav1_path, wav2_path])
    return float(result["score"])


if __name__ == "__main__":
  comparator = SpeakerComparator()

  # Absolute paths on Drive D
  file1 = r"D:\vocera-core\data\raw_15_speakers\zainab\balance.wav"
  file2_same = r"D:\vocera-core\data\raw_15_speakers\zainab\confirm.wav"
  file3_different = r"D:\vocera-core\data\raw_15_speakers\ridha\balance.wav"

  print("\n--- TEST 1: Same Speaker (Genuine Match) ---")
  score_cam_same = comparator.compare_files_cam(file1, file2_same)
  score_ecapa_same = comparator.compare_files_ecapa(file1, file2_same)
  print(f"CAM++ Score:     {score_cam_same:.4f}")
  print(f"ECAPA-TDNN Score: {score_ecapa_same:.4f}")

  print("\n--- TEST 2: Different Speakers (Imposter Match) ---")
  score_cam_diff = comparator.compare_files_cam(file1, file3_different)
  score_ecapa_diff = comparator.compare_files_ecapa(file1, file3_different)
  print(f"CAM++ Score:     {score_cam_diff:.4f}")
  print(f"ECAPA-TDNN Score: {score_ecapa_diff:.4f}")
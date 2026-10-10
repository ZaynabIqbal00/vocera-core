import itertools
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from speechbrain.inference.speaker import EncoderClassifier


class DualEEREvaluator:

  def __init__(
      self,
      data_dir: str = r"D:\vocera-core\data\raw_15_speakers",
      device: str = "cpu",
  ):
    self.data_dir = data_dir
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
    data, fs = sf.read(wav_path)
    signal = torch.from_numpy(data).float()

    if signal.ndim == 1:
      signal = signal.unsqueeze(0)
    elif signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
      signal = signal.T

    if signal.shape[0] > 1:
      signal = torch.mean(signal, dim=0, keepdim=True)

    if fs != 16000:
      resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
      signal = resampler(signal)

    return signal

  def get_ecapa_embedding(self, wav_path: str) -> torch.Tensor:
    signal = self._preprocess_audio(wav_path)
    with torch.no_grad():
      emb = self.ecapa_model.encode_batch(signal)
    return emb.squeeze()

  def build_pair_dataset(self, num_genuine: int = 150, num_imposter: int = 150):
    speakers = [
        s
        for s in os.listdir(self.data_dir)
        if os.path.isdir(os.path.join(self.data_dir, s))
    ]

    speaker_files = {}
    for spk in speakers:
      spk_folder = os.path.join(self.data_dir, spk)
      wavs = [
          os.path.join(spk_folder, f)
          for f in os.listdir(spk_folder)
          if f.endswith(".wav") or f.endswith(".webm")
      ]
      if len(wavs) >= 2:
        speaker_files[spk] = wavs

    genuine_pairs = []
    for spk, files in speaker_files.items():
      for f1, f2 in itertools.combinations(files, 2):
        genuine_pairs.append((f1, f2))

    imposter_pairs = []
    spk_list = list(speaker_files.keys())
    for i in range(len(spk_list)):
      for j in range(i + 1, len(spk_list)):
        spk1, spk2 = spk_list[i], spk_list[j]
        for f1 in speaker_files[spk1]:
          for f2 in speaker_files[spk2]:
            imposter_pairs.append((f1, f2))

    random.seed(42)
    random.shuffle(genuine_pairs)
    random.shuffle(imposter_pairs)

    return genuine_pairs[:num_genuine], imposter_pairs[:num_imposter]

  def compute_scores_ecapa(self, pairs):
    scores = []
    for f1, f2 in pairs:
      emb1 = self.get_ecapa_embedding(f1)
      emb2 = self.get_ecapa_embedding(f2)
      sim = float(F.cosine_similarity(emb1, emb2, dim=0))
      scores.append(sim)
    return np.array(scores)

  def compute_scores_cam(self, pairs):
    scores = []
    for f1, f2 in pairs:
      res = self.cam_pipeline([f1, f2])
      scores.append(float(res["score"]))
    return np.array(scores)

  def calculate_eer(self, genuine_scores, imposter_scores):
    thresholds = np.linspace(-1.0, 1.0, 1000)
    far_list = []
    frr_list = []

    for t in thresholds:
      far = np.mean(imposter_scores >= t)
      frr = np.mean(genuine_scores < t)
      far_list.append(far)
      frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    idx = np.nanargmin(np.abs(far_list - frr_list))
    eer = (far_list[idx] + frr_list[idx]) / 2.0
    optimal_threshold = thresholds[idx]

    return eer, optimal_threshold, thresholds, far_list, frr_list


if __name__ == "__main__":
  evaluator = DualEEREvaluator()

  print("\nGenerating pair combinations across all 15 speakers...")
  genuine_pairs, imposter_pairs = evaluator.build_pair_dataset(
      num_genuine=150, num_imposter=150
  )
  print(
      f"Testing {len(genuine_pairs)} genuine pairs and {len(imposter_pairs)}"
      " imposter pairs..."
  )

  print("\n[1/2] Evaluating ECAPA-TDNN...")
  ecapa_gen = evaluator.compute_scores_ecapa(genuine_pairs)
  ecapa_imp = evaluator.compute_scores_ecapa(imposter_pairs)
  eer_ecapa, thresh_ecapa, threshs, far_ecapa, frr_ecapa = (
      evaluator.calculate_eer(ecapa_gen, ecapa_imp)
  )

  print("\n[2/2] Evaluating CAM++...")
  cam_gen = evaluator.compute_scores_cam(genuine_pairs)
  cam_imp = evaluator.compute_scores_cam(imposter_pairs)
  eer_cam, thresh_cam, _, far_cam, frr_cam = evaluator.calculate_eer(
      cam_gen, cam_imp
  )

  print("\n" + "=" * 55)
  print("   VOCERA COMPARATIVE EVALUATION RESULTS (15 SPEAKERS)")
  print("=" * 55)
  print(
      f"ECAPA-TDNN -> EER: {eer_ecapa * 100:.2f}%  |  Optimal Threshold:"
      f" {thresh_ecapa:.4f}"
  )
  print(
      f"CAM++      -> EER: {eer_cam * 100:.2f}%  |  Optimal Threshold:"
      f" {thresh_cam:.4f}"
  )
  print("=" * 55)

  # Comparative Plot
  plt.figure(figsize=(10, 6))
  plt.plot(
      threshs,
      far_ecapa * 100,
      "r--",
      label=f"ECAPA-TDNN FAR (EER: {eer_ecapa * 100:.1f}%)",
  )
  plt.plot(threshs, frr_ecapa * 100, "b--", label="ECAPA-TDNN FRR")

  plt.plot(
      threshs,
      far_cam * 100,
      "g-",
      linewidth=2,
      label=f"CAM++ FAR (EER: {eer_cam * 100:.1f}%)",
  )
  plt.plot(threshs, frr_cam * 100, "m-", linewidth=2, label="CAM++ FRR")

  plt.xlabel("Cosine Similarity Threshold")
  plt.ylabel("Error Rate (%)")
  plt.title("Vocera Benchmark: CAM++ vs ECAPA-TDNN Comparison")
  plt.legend()
  plt.grid(True)

  save_path = r"D:\vocera-core\eval\eer_comparison.png"
  plt.savefig(save_path)
  print(f"\nSaved comparison plot to: {save_path}")
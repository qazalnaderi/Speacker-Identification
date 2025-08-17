import numpy as np
import librosa
import webrtcvad
import noisereduce as nr
from audiomentations import Compose, LowPassFilter, HighPassFilter, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddBackgroundNoise

sr = 16000
NOISE_DIR = "noises_16k"

augment = Compose([
    HighPassFilter(min_cutoff_freq=280.0, max_cutoff_freq=320.0, p=0.8),
    LowPassFilter(min_cutoff_freq=3300.0, max_cutoff_freq=3600.0, p=0.8),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    AddBackgroundNoise(sounds_path=NOISE_DIR, min_snr_db=3.0, max_snr_db=30.0, min_absolute_rms_db=-51, p=0.7),
])

def preprocess(path, augment_audio):
    sig, _ = librosa.load(path, sr=sr, mono=True)
    sig = nr.reduce_noise(y=sig, sr=sr, prop_decrease=0.8)
    sig = librosa.util.normalize(sig)
    sig = sig.astype(np.float32)

    vad = webrtcvad.Vad(0)
    frame_len = int(sr * 30 / 1000)  
    non_silent = []
    for start in range(0, len(sig), frame_len):
      frame = sig[start:start+frame_len]
      if len(frame) == frame_len:
        frame_int = (frame * 32767).astype(np.int16)
        if vad.is_speech(frame_int.tobytes(), sr):
          non_silent.append(frame)
    if not non_silent:
      return None
    sig = np.concatenate(non_silent)

    segment_samples = 3 * sr
    if len(sig) > segment_samples:
      sig = sig[:segment_samples]
    else:
      pad = np.zeros(segment_samples - len(sig))
      sig = np.concatenate([sig, pad])
    if augment_audio:
      sig = augment(samples=sig.astype(np.float32), sample_rate=sr)


    hop_length = 512
    n_fft = 2048
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, delta, delta2]).T

    mean = feats.mean(axis=0, keepdims=True)
    std  = feats.std(axis=0, keepdims=True) + 1e-8
    feats = ((feats - mean) / std).astype(np.float32)

    TARGET_FRAMES = 300
    T = feats.shape[0]
    if T < TARGET_FRAMES:
        pad = np.zeros((TARGET_FRAMES - T, feats.shape[1]), dtype=np.float32)
        feats = np.vstack([feats, pad])
    else:
        feats = feats[:TARGET_FRAMES, :]

    return feats

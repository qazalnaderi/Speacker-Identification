NOISE_DIR = "noises/ESC-50/audio"
output_dir = "noises_16k"

import librosa
import soundfile as sf
import glob
import os
os.makedirs(output_dir, exist_ok=True)

for file in glob.glob(f"{NOISE_DIR}/*.wav"):
    y, _ = librosa.load(file, sr=16000)
    out_path = os.path.join(output_dir, os.path.basename(file))
    sf.write(out_path, y, 16000)


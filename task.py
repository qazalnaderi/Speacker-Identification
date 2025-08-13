import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import glob
import random
import IPython.display as ipd
import numpy as np
import librosa
import webrtcvad
import noisereduce as nr


path = "raw_data/vox1_dev_wav/wav"

audios = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)

sr = 1600
for _ in range(3):
    file = random.choice(audios)
    print(file)
    sr = 16000
    sig, sr = librosa.load(file, sr=sr, mono=True)
    clean_Sig = nr.reduce_noise(y=sig, sr=sr, prop_decrease=0.8)
    normal_sig = librosa.util.normalize(clean_Sig)
    vad = webrtcvad.Vad(0)

    frame_duration = 30
    frame_length = int(sr * frame_duration / 1000)
    non_silent_fs = []
    for start in range(0, len(normal_sig), frame_length):
        frame = normal_sig[start:start + frame_length]
        if len(frame) == frame_length:
            frame_int = (frame * 32767).astype(np.int16)
            is_speech = vad.is_speech(frame_int.tobytes(), sr)

            if is_speech:
                non_silent_fs.append(frame)

        y_no_silence = np.concatenate(non_silent_fs)
        librosa.display.waveshow(y_no_silence, sr=sr, alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform without Silence")
        plt.show()


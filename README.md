# 🎙️ Speaker Identification with Deep Learning

This project is a deep learning-based **Speaker Identification System** developed for SmartVoice, a speech technology company. It uses the [VoxCeleb1 dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) to identify speakers based on short audio clips, even in noisy conditions or with varied accents.

## 🚀 Features

- 🔊 Extracts audio features: **MFCC**, **Spectrograms**, and **Chroma**
- 🧠 Trains a deep learning model to identify speakers
- 🧹 Handles noisy audio through preprocessing and segmentation
- 📊 Evaluates performance using Accuracy, Precision, Recall, F1-Score
- 📈 Visualizes data using t-SNE, confusion matrix, spectrograms
- 🎯 Clusters speakers by gender, accent, or similarity

---

## 📁 Dataset

We use the [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset:
- 1,251 speakers
- Over 100,000 WAV files
- Includes real-world noise and varied speech quality

You can also test with your own voice or clips from [Freesound.org](https://freesound.org).

---

## 🛠️ Tools & Libraries

- **Python 3.8+**
- `librosa` – audio processing
- `numpy`, `pandas` – data handling
- `matplotlib`, `seaborn` – visualization
- `scikit-learn` – evaluation and clustering
- `PyTorch` – deep learning framework


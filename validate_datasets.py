import os
import librosa
import matplotlib.pyplot as plt
from glob import glob


def plot_sample_distribution():
    pos_count = len(glob("dataset/augmented/*positive*.wav"))
    neg_count = len(glob("dataset/augmented/*negative*.wav"))

    plt.figure(figsize=(10, 5))
    plt.bar(['正样本', '负样本'], [pos_count, neg_count])
    plt.title('数据集分布')
    plt.show()


def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    plt.figure(figsize=(15, 3))
    plt.plot(y)
    plt.title(os.path.basename(audio_path))
    plt.show()


if __name__ == '__main__':
    plot_sample_distribution()
    plot_waveform("dataset/raw/*.wav")

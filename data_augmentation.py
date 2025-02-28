import librosa
import numpy as np
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm


class AudioAugmentor:
    def __init__(self, noise_dir="dataset/background"):
        self.noise_files = glob(os.path.join(noise_dir, "*.wav"))

    def augment(self, audio_path, output_dir, num_augments=5):
        """对单个音频文件执行多种增强"""
        y, sr = librosa.load(audio_path, sr=16000)
        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]

        # 生成多种变体
        variants = [
            self.add_noise(y, sr),
            self.time_shift(y),
            self.pitch_shift(y, sr),
            self.speed_change(y),
            self.volume_scale(y)
        ]

        # 保存增强结果
        for i, variant in enumerate(variants[:num_augments]):
            output_path = os.path.join(output_dir, f"{base_name}_aug{i}.wav")
            sf.write(output_path, variant, sr)

    def add_noise(self, y, sr, snr=15):
        """添加背景噪声"""
        noise = self._load_random_noise(len(y), sr)
        noise = noise * np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
        return y + noise

    def time_shift(self, y, max_shift=0.2):
        """时间偏移"""
        shift = np.random.randint(int(len(y) * max_shift))
        return np.roll(y, shift)

    def pitch_shift(self, y, sr, semitones=(-3, 3)):
        """音高变化（针对'zh'音优化）"""
        n_steps = np.random.uniform(*semitones)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def speed_change(self, y, speed_range=(0.9, 1.1)):
        """语速变化"""
        speed = np.random.uniform(*speed_range)
        return librosa.effects.time_stretch(y, rate=speed)

    def volume_scale(self, y, db_range=(-6, 6)):
        """音量缩放"""
        db = np.random.uniform(*db_range)
        return y * 10 ** (db / 20)

    def _load_random_noise(self, length, sr):
        """加载随机背景噪声"""
        noise_path = np.random.choice(self.noise_files)
        noise, _ = librosa.load(noise_path, sr=sr)
        if len(noise) < length:
            noise = np.tile(noise, (length // len(noise)) + 1)
        start = np.random.randint(0, len(noise) - length)
        return noise[start:start + length]


if __name__ == '__main__':
    aug = AudioAugmentor()
    raw_files = glob("dataset/raw/*.wav")

    os.makedirs("dataset/augmented", exist_ok=True)
    for file in tqdm(raw_files):
        aug.augment(file, "dataset/augmented")

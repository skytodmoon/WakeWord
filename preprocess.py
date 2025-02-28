import librosa
import tensorflow as tf
import numpy as np
import os
from glob import glob

def extract_features(file_path, num_mfcc=31, n_fft=512, hop_length=256):
    """提取MFCC特征并确保形状一致"""
    y, sr = librosa.load(file_path, sr=16000)

    # 统一音频长度为1.5秒
    target_length = 1.5 * sr  # 24000 samples
    if len(y) < target_length:
        y = np.pad(y, (0, int(target_length - len(y))), mode='constant')
    else:
        y = y[:int(target_length)]

    # 提取MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    # 确保时间轴长度为30帧
    if mfcc.shape[1] < 30:
        mfcc = np.pad(mfcc, ((0, 0), (0, 30 - mfcc.shape[1])), mode='edge')
    else:
        mfcc = mfcc[:, :30]

    return mfcc.T  # 转换为 (时间帧, 特征数)


def create_tfrecord(input_dir, output_dir, label):
    """创建TFRecord数据集"""
    os.makedirs(output_dir, exist_ok=True)
    writer = tf.io.TFRecordWriter(os.path.join(output_dir, f"data_{label}.tfrecord"))

    for file in glob(os.path.join(input_dir, "*.wav")):
        try:
            mfcc = extract_features(file)
            assert mfcc.shape == (30, 31), f"特征形状错误: {mfcc.shape}"

            example = tf.train.Example(features=tf.train.Features(feature={
                'feature': tf.train.Feature(float_list=tf.train.FloatList(value=mfcc.flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())
        except Exception as e:
            print(f"处理{file}失败: {str(e)}")

    writer.close()


if __name__ == "__main__":
    # 处理正样本（唤醒词）
    create_tfrecord(
        input_dir="dataset/augmented/positive",
        output_dir="dataset/train",
        label=1
    )

    # 处理负样本
    create_tfrecord(
        input_dir="dataset/augmented/negative",
        output_dir="dataset/train",
        label=0
    )

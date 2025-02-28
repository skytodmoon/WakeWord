# train_model.py
import tensorflow as tf
import numpy as np
import os
from glob import glob


# ======================
# 配置参数
# ======================
class Config:
    BATCH_SIZE = 64
    EPOCHS = 50
    INPUT_SHAPE = (30, 31, 1)  # MFCC特征维度
    SAMPLE_RATE = 16000
    NUM_MFCC = 31
    NUM_FRAMES = 30


# ======================
# Metal GPU配置
# ======================
def configure_metal():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 配置内存增长和显存限制
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
            print(f"✅ Metal GPU已配置 | 显存限制: 4096MB")
        except RuntimeError as e:
            print("⚠️ 配置错误:", e)


# ======================
# 数据管道（关键修正）
# ======================
def build_data_pipeline():
    def _parse_function(example_proto):
        # 修正特征解析描述
        feature_description = {
            'feature': tf.io.FixedLenFeature([Config.NUM_FRAMES * Config.NUM_MFCC], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['feature'], parsed['label']

    def _preprocess(feature, label):
        # 先reshape再类型转换
        feature = tf.reshape(feature, (Config.NUM_FRAMES, Config.NUM_MFCC, 1))
        return tf.cast(feature, tf.float32), tf.cast(label, tf.int32)

    # 构建训练管道
    train_ds = tf.data.TFRecordDataset(glob("dataset/train/*.tfrecord"))
    train_ds = (
        train_ds
        .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(Config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 构建验证管道
    val_ds = tf.data.TFRecordDataset(glob("dataset/val/*.tfrecord"))
    val_ds = (
        val_ds
        .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(Config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# ======================
# 模型架构（适配输入维度）
# ======================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=Config.INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ======================
# 数据验证（新增）
# ======================
def validate_data():
    # 验证单个样本
    sample_path = glob("dataset/train/*.tfrecord")[0]
    sample_ds = tf.data.TFRecordDataset(sample_path).take(1)

    for raw_record in sample_ds:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        feature = np.array(example.features.feature['feature'].float_list.value)
        assert len(feature) == Config.NUM_FRAMES * Config.NUM_MFCC, \
            f"❌ 特征维度错误: 预期{Config.NUM_FRAMES * Config.NUM_MFCC} 实际{len(feature)}"

    # 验证预处理管道
    train_ds, _ = build_data_pipeline()
    sample_batch = next(iter(train_ds))
    features, labels = sample_batch
    assert features.shape[1:] == Config.INPUT_SHAPE, \
        f"❌ 输入形状错误: 预期{Config.INPUT_SHAPE} 实际{features.shape[1:]}"
    print("✅ 数据验证通过")


# ======================
# 训练流程
# ======================
def main():
    configure_metal()
    validate_data()

    train_ds, val_ds = build_data_pipeline()
    model = build_model()

    # 训练回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
    ]

    # 开始训练
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )

    # 保存最终模型
    model.save("/models/xiaozhi_detector.keras")
    print("🎉 训练完成！")


if __name__ == "__main__":
    main()

# 避免依赖JAX的转换方式
import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model('xiaozhi.h5')
tfjs.converters.save_keras_model(
    model,
    'static/web_model',
    quantization_dtype='float16',
    weight_shard_size_bytes=4*1024*1024
)
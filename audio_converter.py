from pydub import AudioSegment
import os
import glob


def batch_convert_to_wav(input_dir, output_dir, target_sr=16000):
    """
    批量转换文件夹内的音频文件为标准WAV格式
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param target_sr: 目标采样率（默认16000Hz）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 支持多种常见音频格式
    audio_extensions = ['*.wav', '*.mp3', '*.ogg', '*.flac', '*.m4a']

    # 递归遍历所有子目录
    for root, _, _ in os.walk(input_dir):
        # 处理所有支持的音频格式
        for ext in audio_extensions:
            for audio_path in glob.glob(os.path.join(root, ext)):
                # 构建输出路径（保留原始目录结构）
                rel_path = os.path.relpath(audio_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_path = os.path.splitext(output_path)[0] + '.wav'

                # 创建子目录（如果不存在）
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # 执行格式转换
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_frame_rate(target_sr)
                    audio = audio.set_channels(1)
                    audio.export(output_path,
                                 format='wav',
                                 codec='pcm_s16le',
                                 parameters=['-ar', str(target_sr)])
                    print(f"转换成功: {audio_path} -> {output_path}")
                except Exception as e:
                    print(f"转换失败: {audio_path} - 错误: {str(e)}")


# 使用示例
if __name__ == '__main__':
    # input_folder = "dataset/raw"
    # output_folder = "dataset/converted"
    #
    # batch_convert_to_wav(
    #     input_dir=input_folder,
    #     output_dir=output_folder,
    #     target_sr=16000
    # )
    input_folder = "dataset/background"
    output_folder = "dataset/converted"

    batch_convert_to_wav(
        input_dir=input_folder,
        output_dir=output_folder,
        target_sr=16000
    )

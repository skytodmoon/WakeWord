from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'dataset/raw'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 确保静态文件夹存在
os.makedirs('static', exist_ok=True)
os.makedirs('static/web_model', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        user_id = request.form.get('user_id', 'anonymous')
        audio_type = request.form['type']  # 'positive'/'negative'
        audio_file = request.files['audio']

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{user_id}_{audio_type}_{timestamp}.wav"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        audio_file.save(save_path)
        return jsonify({'status': 'success', 'path': save_path})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 添加语音识别API代理路由
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_proxy():
    try:
        # 这里只是一个模拟，实际项目中应该调用真实的API
        # 例如百度语音识别、讯飞等
        return jsonify({
            'status': 'success',
            'text': '这是一个模拟的语音识别结果',
            'confidence': 0.95
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

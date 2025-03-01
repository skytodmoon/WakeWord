// app.js
class WakeWordDetector {
    constructor() {
        this.model = null;
        this.vad = null;
        this.isListening = false;
        this.isCommandMode = false;  // 新增：指令模式标志
        this.commandAudioBuffer = []; // 新增：指令音频缓冲区
        this.apiKey = 'YOUR_API_KEY'; // 新增：API密钥
        this.audioContext = null;
        this.scriptProcessor = null;
        this.bufferSize = 1024;
        this.sampleRate = 16000;
    }

    async init() {
        // 加载模型
        this.model = await tf.loadGraphModel('static/web_model/model.json');

        // 初始化VAD
        this.vad = new VadJs.VAD({
            workaroundUserMedia: true,
            minEnergy: 0.0015,  // 调整此参数以适应环境噪音
            bufferSize: this.bufferSize
        });

        document.getElementById('startBtn').addEventListener('click', () => this.start());
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
            const source = this.audioContext.createMediaStreamSource(stream);

            // 创建音频处理节点
            this.scriptProcessor = this.audioContext.createScriptProcessor(
                this.bufferSize, 1, 1
            );

            // 连接处理链路
            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);

            // 注册处理回调
            this.scriptProcessor.onaudioprocess = (e) => this.processAudio(e);
            this.isListening = true;
            this.updateStatus('监听中...');
        } catch (err) {
            console.error('麦克风访问失败:', err);
        }
    }

    processAudio(event) {
        const input = event.inputBuffer.getChannelData(0);

        // 如果在指令收集模式，直接保存音频
        if (this.isCommandMode) {
            this.collectCommandAudio(input);
            return;
        }

        // VAD检测
        const isSpeech = this.vad.process(input);
        if (!isSpeech) return;

        // 提取MFCC特征
        const mfcc = this.extractMFCC(input);

        // 模型推断
        const tensor = tf.tensor4d(mfcc, [1, 30, 31, 1]);
        const prediction = this.model.predict(tensor);
        const prob = prediction.dataSync()[0];

        if (prob > 0.85) {  // 置信度阈值
            this.triggerWakeWord();
        }
    }

    // 新增：收集指令音频
    collectCommandAudio(audioData) {
        // 将音频数据添加到缓冲区
        this.commandAudioBuffer.push(...audioData);
    }

    // 修改：唤醒词触发函数
    triggerWakeWord() {
        this.updateStatus('🚨 检测到唤醒词 "小智"!');
        
        // 延迟一小段时间，让用户准备说指令
        setTimeout(() => {
            this.startCommandRecognition();
        }, 500);
    }

    // 新增：开始指令识别
    startCommandRecognition() {
        // 切换到指令模式
        this.isCommandMode = true;
        this.commandAudioBuffer = [];
        this.updateStatus('请说出您的指令...');
        
        // 设置超时，防止用户不说话
        this.commandTimeout = setTimeout(() => {
            if (this.isCommandMode) {
                this.endCommandRecognition();
                this.updateStatus('未检测到指令，已返回唤醒词监听');
            }
        }, 5000);
        
        // 设置VAD回调以检测语音结束
        this.setupVADForCommand();
    }
    
    // 新增：设置VAD用于指令检测
    setupVADForCommand() {
        // 保存原始VAD配置
        const originalProcess = this.vad.process;
        
        // 重写VAD处理函数
        this.vad.process = (input) => {
            const isSpeech = originalProcess.call(this.vad, input);
            
            // 如果检测到语音并且是指令模式
            if (this.isCommandMode) {
                if (isSpeech) {
                    // 重置超时
                    clearTimeout(this.commandTimeout);
                    this.lastSpeechTime = Date.now();
                } else if (this.lastSpeechTime && Date.now() - this.lastSpeechTime > 1500) {
                    // 如果1.5秒没有语音，认为指令结束
                    this.endCommandRecognition();
                }
            }
            
            return isSpeech;
        };
    }
    
    // 新增：结束指令识别
    async endCommandRecognition() {
        if (!this.isCommandMode) return;
        
        this.isCommandMode = false;
        clearTimeout(this.commandTimeout);
        this.updateStatus('正在处理指令...');
        
        // 准备音频数据
        const audioBlob = this.prepareAudioBlob();
        
        try {
            // 发送到API
            const result = await this.sendToSpeechAPI(audioBlob);
            
            // 显示识别结果
            this.updateStatus(`识别结果: ${result.text}`);
            console.log('完整识别结果:', result);
            
            // 这里可以添加指令处理逻辑
            
        } catch (error) {
            console.error('语音识别失败:', error);
            this.updateStatus('语音识别失败');
        }
        
        // 3秒后恢复唤醒词监听
        setTimeout(() => {
            this.updateStatus('监听中...');
        }, 3000);
    }
    
    // 新增：准备音频Blob
    prepareAudioBlob() {
        // 将Float32Array转换为16位PCM
        const pcmBuffer = new Int16Array(this.commandAudioBuffer.length);
        for (let i = 0; i < this.commandAudioBuffer.length; i++) {
            pcmBuffer[i] = Math.min(1, Math.max(-1, this.commandAudioBuffer[i])) * 0x7FFF;
        }
        
        // 创建WAV文件头
        const wavHeader = this.createWavHeader(pcmBuffer.length * 2);
        
        // 合并头和数据
        const wavBuffer = new Uint8Array(wavHeader.length + pcmBuffer.length * 2);
        wavBuffer.set(wavHeader);
        
        // 复制PCM数据
        const pcmView = new DataView(wavBuffer.buffer, wavHeader.length);
        for (let i = 0; i < pcmBuffer.length; i++) {
            pcmView.setInt16(i * 2, pcmBuffer[i], true);
        }
        
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }
    
    // 新增：创建WAV头
    createWavHeader(dataLength) {
        const buffer = new ArrayBuffer(44);
        const view = new DataView(buffer);
        
        // RIFF标识
        view.setUint8(0, 'R'.charCodeAt(0));
        view.setUint8(1, 'I'.charCodeAt(0));
        view.setUint8(2, 'F'.charCodeAt(0));
        view.setUint8(3, 'F'.charCodeAt(0));
        
        // 文件长度
        view.setUint32(4, 36 + dataLength, true);
        
        // WAVE标识
        view.setUint8(8, 'W'.charCodeAt(0));
        view.setUint8(9, 'A'.charCodeAt(0));
        view.setUint8(10, 'V'.charCodeAt(0));
        view.setUint8(11, 'E'.charCodeAt(0));
        
        // fmt子块
        view.setUint8(12, 'f'.charCodeAt(0));
        view.setUint8(13, 'm'.charCodeAt(0));
        view.setUint8(14, 't'.charCodeAt(0));
        view.setUint8(15, ' '.charCodeAt(0));
        
        // 子块长度
        view.setUint32(16, 16, true);
        // 音频格式 (PCM = 1)
        view.setUint16(20, 1, true);
        // 通道数
        view.setUint16(22, 1, true);
        // 采样率
        view.setUint32(24, this.sampleRate, true);
        // 字节率
        view.setUint32(28, this.sampleRate * 2, true);
        // 块对齐
        view.setUint16(32, 2, true);
        // 每个样本位数
        view.setUint16(34, 16, true);
        
        // data子块
        view.setUint8(36, 'd'.charCodeAt(0));
        view.setUint8(37, 'a'.charCodeAt(0));
        view.setUint8(38, 't'.charCodeAt(0));
        view.setUint8(39, 'a'.charCodeAt(0));
        
        // 数据长度
        view.setUint32(40, dataLength, true);
        
        return new Uint8Array(buffer);
    }
    
    // 新增：发送到语音识别API
    async sendToSpeechAPI(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        formData.append('language', 'zh-CN');
        
        const response = await fetch('https://api.example.com/speech-to-text', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        return await response.json();
    }

    extractMFCC(audioData) {
        // 简化的浏览器端MFCC提取（实际需完整实现）
        const frameSize = 512;
        const hopSize = 256;
        const mfcc = [];

        // 分帧处理
        for (let i = 0; i < audioData.length; i += hopSize) {
            const frame = audioData.slice(i, i + frameSize);
            // 此处应实现完整的MFCC计算逻辑
            // 可使用https://github.com/audiojs/mfcc等库
            const features = this.calculateMFCC(frame);
            mfcc.push(features);
        }

        return mfcc.slice(0, 30);  // 截取前30帧
    }

    calculateMFCC(frame) {
        // 简化的MFCC计算示例（需完整实现）
        const fft = new FFT(frame.length);
        const spectrum = fft.createSpectrum(frame);
        const melBands = this.applyMelFilterbank(spectrum);
        return this.dct(melBands);
    }

    triggerWakeWord() {
        this.updateStatus('🚨 检测到唤醒词 "小智"!');
        // 触发后续操作，如语音识别等
    }

    updateStatus(text) {
        document.getElementById('status').textContent = `状态: ${text}`;
    }
}

// 初始化检测器
const detector = new WakeWordDetector();
detector.init();

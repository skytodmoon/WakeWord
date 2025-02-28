// app.js
class WakeWordDetector {
    constructor() {
        this.model = null;
        this.vad = null;
        this.isListening = false;
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

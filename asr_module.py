import vosk
vosk.SetLogLevel(-1)  # 屏蔽日志输出
import pyaudio
import json

class asr_module():
    def __init__(self):
        # 语音指令集
        self.COMMANDS = '["上", "下", "左", "右", "前", "后", "顺时针", "逆时针", "起飞", "悬停", "降落", "断电"]'
        # 注意这会大幅提高召回率但降低准确率
        # 也就是说你说的所有其他话都可能被误识别成命令，命令之间有一定概率被互相误识别，需要先熟悉模型的发音
        
        # 部署一个轻量本地模型，似乎是在CPU上
        self.model = vosk.Model("resource/vosk-model-small-cn-0.22")
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000, self.COMMANDS)#16000是比特率

        # 监测音频输入
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                            input=True, frames_per_buffer=3200)
        
        print("\n语音识别模块初始化成功")

    def get_result(self):
        '''
        返回当前识别到的语音指令
        '''
        data = self.stream.read(2000, exception_on_overflow=False)
        self.recognizer.AcceptWaveform(data)
        pr = json.loads(self.recognizer.PartialResult()).get('partial','')
        if pr != '':
            # 靠获取FinalResult强制跳过静音延迟等待
            self.recognizer.FinalResult()

        return pr


if __name__ == '__main__':
    Audio_Recognizer = asr_module()
    while True:
        command = Audio_Recognizer.get_result()
        if command != '':
            print(command)



